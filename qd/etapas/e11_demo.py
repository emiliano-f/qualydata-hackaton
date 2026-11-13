"""e11 · Tablero de demo (`reportes/<set>_demo.html`): qué circula, qué apareció, quién empuja.

Tablero de presentación separado del reporte e10 (que no se toca). Lee tablas que
ya existen en el almacén (com, asig, narr, grupos, grupos_diag, coord_pares,
coord_pares_todos, pub, cta) y el costo de la corrida: NO escribe en el DuckDB ni
en entrega/. Solo presentación: todo número sale de esas tablas.

Robustez: solo stdlib a nivel de módulo; pandas/numpy/networkx/plotly se importan
dentro de run(). Cualquier fallo queda en log.warning y el pipeline sigue.

Standalone (para generar desde copias de otro clon):
    python -m qd.etapas.e11_demo --db RUTA/qd.duckdb --costo RUTA/costo.json \
        --set final --out reportes/final_demo.html
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("qd")
SALIDAS: list = []
TZ = timezone(timedelta(hours=-3))

# Paul Tol "muted"; el 2º original #332288 no contrasta sobre #0b0d12 -> #6a5acd.
PALETA = ["#CC6677", "#6a5acd", "#DDCC77", "#117733", "#88CCEE",
          "#882255", "#44AA99", "#999933", "#AA4499"]
GRIS = "#6b7080"
COLORES_GRUPO = ["#e8e8e8", "#f5d76e", "#4dd8e8", "#f0a0c8", "#a8d878", "#c898f0"]
PASOS = ["Circula", "Apareció", "Empujan", "Lo sabemos", "Costo"]


_MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def _f(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=TZ).strftime("%Y-%m-%d %H:%M")


def _fecha(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=TZ).strftime("%Y-%m-%d")


def _fcorta(ts) -> str:
    d = datetime.fromtimestamp(int(ts), tz=TZ)
    return f"{d.day}-{_MESES[d.month - 1]}"


def _esc(x) -> str:
    return html.escape(str(x))


def _num(x, dec=None) -> str:
    return (f"{x:,.{dec}f}" if dec is not None else f"{x:,}").replace(",", " ").replace(".", ",").replace(" ", ".")


def _usd(x) -> str:
    """es-AR: 2 decimales si >= 0,01; si no, 2 cifras significativas; 0 exacto -> '0'."""
    x = float(x or 0)
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return _num(x, 2)
    return _num(float(f"{x:.2g}"), 6)


def _plural(n: int, sing: str, plur: str) -> str:
    return f"{n} {sing if n == 1 else plur}"


def _recortar(texto: str, limite: int) -> str:
    t = " ".join(str(texto).split())
    if len(t) <= limite:
        return t
    return t[:limite].rsplit(" ", 1)[0] + "…"


def _etiqueta(g):
    """Comentario real más representativo: mayor conf (coseno al centroide del micro, e03),
    con >= 25 caracteres, deduplicado por texto, desempate por comentario_id."""
    sub = g[g["texto"].str.len() >= 25].drop_duplicates("texto")
    if not len(sub):
        sub = g.drop_duplicates("texto")
    sub = sub.sort_values(["conf", "comentario_id"], ascending=[False, True])
    return _recortar(sub["texto"].iloc[0], 70) if len(sub) else ""


def recolectar(db, cfg, costo, nombre_set: str) -> dict:
    """Toda la agregación; devuelve el payload serializable para el HTML."""
    import numpy as np
    import pandas as pd

    from qd.config import obtener
    from qd.etapas.e10_reporte import (hashtags_raros, mapa_hashtags, posts_compartidos,
                                     respuestas_mutuas)

    com = db.leer("com")[["comentario_id", "cuenta_id", "publicacion_id", "ts", "texto", "respuesta_a"]]
    asig = db.leer("asig")[["comentario_id", "narrativa", "postura", "conf"]]
    narr = (db.leer("narr") if db.existe("narr")
            else pd.DataFrame(columns=["narrativa", "titulo", "frases_clave", "ts_origen", "n"]))
    grupos = db.leer("grupos") if db.existe("grupos") else pd.DataFrame(columns=["cuenta_id", "grupo"])
    grupos = grupos[grupos["cuenta_id"] != ""]
    gdiag = db.leer("grupos_diag") if db.existe("grupos_diag") else pd.DataFrame()
    gdiag = gdiag[gdiag["grupo"] != ""] if len(gdiag) else gdiag
    pub = db.leer("pub") if db.existe("pub") else pd.DataFrame(columns=["publicacion_id", "titular"])
    cta = db.leer("cta") if db.existe("cta") else pd.DataFrame(columns=["cuenta_id", "usuario"])
    cp = (db.leer("coord_pares") if db.existe("coord_pares")
          else pd.DataFrame(columns=["u", "v", "lift"]))
    cpt = (db.leer("coord_pares_todos") if db.existe("coord_pares_todos")
           else pd.DataFrame(columns=["u", "v"]))

    df = com.merge(asig, on="comentario_id").merge(grupos, on="cuenta_id", how="left")
    df["grupo"] = df["grupo"].fillna("")
    nar = df[df["narrativa"] != ""]
    t0, t1 = int(com["ts"].min()), int(com["ts"].max())
    span = max(t1 - t0, 1)

    # --- narrativas mostradas (mismo filtro que e10), emergentes y colores
    conteo = nar["narrativa"].value_counts()
    min_frac = float(obtener(cfg, "reporte.min_frac", 0.001))
    orden = conteo[conteo >= min_frac * len(com)].index.tolist()
    titulos = dict(zip(narr["narrativa"], narr["titulo"])) if len(narr) else {}
    frases = (dict(zip(narr["narrativa"], narr["frases_clave"]))
              if len(narr) and "frases_clave" in narr else {})
    idx_o = narr.set_index("narrativa")["ts_origen"] if len(narr) else pd.Series(dtype=float)
    ts_o, emergente = {}, {}
    for lab in orden:
        g = nar[nar["narrativa"] == lab]
        o = (int(idx_o[lab]) if lab in idx_o.index and pd.notna(idx_o[lab]) else int(g["ts"].min()))
        ts_o[lab] = o
        emergente[lab] = (o - t0) > 0.1 * span
    emerg = [lab for lab in orden if emergente[lab]]
    coloreadas = orden[:len(PALETA)]
    for lab in orden[len(PALETA):]:
        if not emergente[lab]:
            continue
        for j in range(len(coloreadas) - 1, -1, -1):     # la emergente toma el slot
            if not emergente[coloreadas[j]]:             # de la más chica no emergente
                coloreadas[j] = lab
                break
    color_n = {lab: coloreadas.index(lab) % len(PALETA) if lab in coloreadas else -1
               for lab in orden}

    # --- filas de narrativas (etiqueta = comentario más central)
    filas = []
    for lab in orden:
        g = nar[nar["narrativa"] == lab]
        pct = g["postura"].value_counts(normalize=True)
        ej = (g.sort_values("conf", ascending=False).drop_duplicates("texto")["texto"]
              .head(3).map(lambda s: _recortar(s, 160)).tolist())
        filas.append({"id": lab, "etiqueta": _etiqueta(g), "titulo": titulos.get(lab, lab),
                      "frases": str(frases.get(lab, "")).replace("|", " · "), "i": color_n[lab],
                      "n": int(len(g)), "apoya": float(pct.get("apoya", 0.0)),
                      "rechaza": float(pct.get("rechaza", 0.0)), "origen": _f(ts_o[lab]),
                      "origen_fecha": _fcorta(ts_o[lab]), "emergente": emergente[lab],
                      "coord": float((g["grupo"] != "").mean()), "ej": ej})
    max_dia = {}
    nar_v = nar[nar["narrativa"].isin(orden)].copy()
    nar_v["b"] = (nar_v["ts"] - t0) // 86400
    nb = int((t1 - t0) // 86400) + 1
    x = [_fecha(t0 + i * 86400) for i in range(nb)]
    tab = nar_v.groupby(["b", "narrativa"]).size().unstack(fill_value=0).reindex(range(nb), fill_value=0)
    serie_dia = {lab: (tab[lab].astype(int).tolist() if lab in tab else [0] * nb) for lab in orden}
    for lab in emerg:
        max_dia[lab] = max(serie_dia[lab]) if serie_dia.get(lab) else 0

    # --- tarjetas de grupos
    delta = int(obtener(cfg, "coord.delta_s", 450))
    umbral_tag = max(int(obtener(cfg, "coord.min_size", 5)),
                     float(obtener(cfg, "coord.hashtag_max_frac", 0.01)) * df["cuenta_id"].nunique())
    uso_total = mapa_hashtags(df[["cuenta_id", "texto"]]) if len(grupos) else {}
    diag = gdiag.set_index("grupo") if len(gdiag) else pd.DataFrame()
    horas_ng = [0] * 24
    if len(grupos):
        ng = df[df["grupo"] == ""]
        tot_ng = max(len(ng), 1)
        for h, c in (((ng["ts"] - 3 * 3600) // 3600) % 24).value_counts().items():
            horas_ng[int(h)] = round(100 * c / tot_ng, 2)
    titular = dict(zip(pub["publicacion_id"], pub["titular"])) if len(pub) else {}
    tarjetas = []
    for gid in grupos["grupo"].value_counts().index.tolist():
        ctas = set(grupos.loc[grupos["grupo"] == gid, "cuenta_id"])
        g = df[df["cuenta_id"].isin(ctas)]
        gn = g[g["narrativa"] != ""]
        empuja = [{"id": lab, "etiqueta": next((r["etiqueta"] for r in filas if r["id"] == lab), lab),
                   "i": color_n.get(lab, -1), "n": int(cnt), "pct": 100 * cnt / max(len(g), 1)}
                  for lab, cnt in gn["narrativa"].value_counts().head(3).items()]
        nx_txt = "-"
        if len(diag) and gid in diag.index:
            co, cn = diag.loc[gid, "coh_obs"], diag.loc[gid, "coh_nulo"]
            if pd.notna(co) and pd.notna(cn):
                nx_txt = f"{co / cn:.1f}×".replace(".", ",") if cn else "mucho más"
        horas = [0] * 24
        for h, c in (((g["ts"] - 3 * 3600) // 3600) % 24).value_counts().items():
            horas[int(h)] = round(100 * c / max(len(g), 1), 2)
        # post con más miembros del grupo, minuto a minuto
        miembros_por_post = g.groupby("publicacion_id")["cuenta_id"].nunique()
        post = {"titular": "", "pts": []}
        if len(miembros_por_post):
            pid = miembros_por_post.idxmax()
            gp = g[g["publicacion_id"] == pid].sort_values("ts")
            if len(gp):
                t_p = int(gp["ts"].iloc[0])
                post = {"titular": str(titular.get(pid, pid)),
                        "pts": [{"m": round((int(r.ts) - t_p) / 60, 1), "c": r.cuenta_id}
                                for r in gp.itertuples()]}
        ej_g = (gn.sort_values("conf", ascending=False).drop_duplicates("texto")["texto"]
                .head(3).map(lambda s: _recortar(s, 160)).tolist())
        tarjetas.append({"id": gid, "n": len(ctas), "nx": nx_txt, "nx_v":
                         (float(diag.loc[gid, "coh_obs"] / diag.loc[gid, "coh_nulo"])
                          if len(diag) and gid in diag.index and diag.loc[gid, "coh_nulo"] else 0.0),
                         "empuja": empuja, "horas": horas,
                         "posts": posts_compartidos(g[["cuenta_id", "publicacion_id", "ts"]], delta),
                         "respuestas": respuestas_mutuas(g[["comentario_id", "cuenta_id", "respuesta_a"]]),
                         "tags": hashtags_raros(mapa_hashtags(g[["cuenta_id", "texto"]]),
                                                uso_total, umbral_tag)[:5],
                         "n_tags": len(hashtags_raros(mapa_hashtags(g[["cuenta_id", "texto"]]),
                                                      uso_total, umbral_tag)),
                         "post": post, "ej": ej_g,
                         "top_pct": float(empuja[0]["pct"]) if empuja else 0.0,
                         "top_lab": empuja[0]["id"] if empuja else "",
                         "top_et": empuja[0]["etiqueta"] if empuja else ""})

    # --- grafo de cuentas: un spring_layout(seed 42) por grupo, ubicado en un centro fijo
    # (círculo ordenado por id de grupo); la muestra de normales ocupa el fondo con el suyo.
    import math

    import networkx as nx

    coord_ctas = sorted(set(grupos["cuenta_id"])) if len(grupos) else []
    gid_de = dict(zip(grupos["cuenta_id"], grupos["grupo"]))
    gids = sorted(set(grupos["grupo"])) if len(grupos) else []
    resto = sorted(set(cta["cuenta_id"]) - set(coord_ctas))
    rng = np.random.default_rng(42)
    muestra = sorted(rng.choice(resto, size=min(300, len(resto)), replace=False).tolist()) if resto else []
    nodos = coord_ctas + muestra
    idx_nodo = {c: i for i, c in enumerate(nodos)}
    aristas = set()
    if len(cp):
        for u, v in cp[["u", "v"]].itertuples(index=False):
            if u in idx_nodo and v in idx_nodo:
                aristas.add(tuple(sorted((u, v))))
    if len(cpt):
        for u, v in cpt[["u", "v"]].itertuples(index=False):
            if u in idx_nodo and v in idx_nodo:
                aristas.add(tuple(sorted((u, v))))
    aristas = sorted(aristas)

    def _layout(sub_nodos, sub_aristas, centro, radio):
        if not sub_nodos:
            return {}
        if len(sub_nodos) == 1:
            return {sub_nodos[0]: np.array(centro, dtype=float)}
        g = nx.Graph()
        g.add_nodes_from(sub_nodos)
        g.add_edges_from(a for a in sub_aristas if a[0] in sub_nodos and a[1] in sub_nodos)
        p = nx.spring_layout(g, seed=42)
        pts = np.array([p[c] for c in sub_nodos])
        esc = float(np.abs(pts).max()) or 1.0
        return {c: np.array(centro) + p[c] / esc * radio for c in sub_nodos}

    ng = max(len(gids), 1)
    centros = {gid: (1.35 * math.cos(2 * math.pi * k / ng - math.pi / 2),
                     1.35 * math.sin(2 * math.pi * k / ng - math.pi / 2))
               for k, gid in enumerate(gids)}
    pos = {}
    for gid in gids:
        pos.update(_layout(sorted(c for c in coord_ctas if gid_de[c] == gid),
                           aristas, centros[gid], 0.42))
    pos.update(_layout(muestra, aristas, (0.0, 0.0), 1.9))
    tip_de = {}
    if len(grupos):
        ej_cta = (df[df["cuenta_id"].isin(coord_ctas)]
                  .sort_values("conf", ascending=False).drop_duplicates("cuenta_id"))
        tip_de = {r.cuenta_id: _recortar(r.texto, 120) for r in ej_cta.itertuples()}
    g_nodos = [{"c": c, "g": gid_de.get(c, ""), "x": round(float(pos[c][0]), 6),
                "y": round(float(pos[c][1]), 6), "tip": tip_de.get(c, "")} for c in nodos]
    grafo = {"nodos": g_nodos, "aristas": [[idx_nodo[u], idx_nodo[v]] for u, v in aristas]}

    dias = max(1, round(span / 86400))
    return {
        "set": nombre_set, "titulo_set": {"practica": "práctica", "final": "set final"}.get(nombre_set, nombre_set),
        "total": len(com), "dias": dias, "n_narr": len(orden), "n_emerg": len(emerg),
        "n_grupos": int(grupos["grupo"].nunique()) if len(grupos) else 0,
        "n_coord": int(len(grupos)), "ruido": float((df["narrativa"] == "").mean()),
        "periodo": f"{_fcorta(t0)} → {_fcorta(t1)}", "x": x, "serie_dia": serie_dia,
        "filas": filas, "emerg": emerg, "origenes": {lab: _f(ts_o[lab]) for lab in emerg},
        "max_dia": max_dia, "tarjetas": tarjetas, "horas_ng": horas_ng,
        "grafo": grafo, "costo": costo,
        "caliente": "e01_ingesta" not in (costo.get("desglose_segundos") or {}),
        "etapas": costo.get("desglose_segundos") or {},
    }


def _frase1(d: dict) -> str:
    s = (f"En {d['dias']} días circularon {_plural(d['n_narr'], 'narrativa', 'narrativas')}; "
         f"{d['n_emerg']} aparecieron de golpe y {_plural(d['n_grupos'], 'grupo coordinado', 'grupos coordinados')} "
         f"({_plural(d['n_coord'], 'cuenta', 'cuentas')}) actúan en bloque.")
    cand = [t for t in d["tarjetas"] if t["top_pct"] >= 50 and t["top_et"]]
    if cand:
        t = max(cand, key=lambda t: (t["nx_v"], t["top_pct"]))
        s += f' El grupo {t["id"]} concentra el {t["top_pct"]:.0f} % de su actividad en «{_esc(t["top_et"])}».'
    return s


def render(d: dict) -> str:
    from qd.etapas.e10_reporte import plotly_js

    payload = json.dumps(d, ensure_ascii=False, default=str)
    n_max = max(1, d["filas"][0]["n"]) if d["filas"] else 1
    filas_p1 = []
    for r in d["filas"]:
        badge = ' <span class="badge">EMERGENTE</span>' if r["emergente"] else ""
        pos_txt = (f"postura: {100 * r['apoya']:.0f} % apoya · {100 * r['rechaza']:.0f} % rechaza · "
                   f"{100 * (1 - r['apoya'] - r['rechaza']):.0f} % sin bando")
        filas_p1.append(f'''<div class="barr" data-nar="{_esc(r["id"])}" data-emerg="{1 if r["emergente"] else 0}"
  data-n="{r["n"]}" data-ej="{_esc(json.dumps(r["ej"][:2]))}" data-pos="{_esc(pos_txt)}">
  <span class="sw" style="background:{'var(--c' + str(r["i"] + 1) + ')' if r["i"] >= 0 else 'var(--otras)'}"></span>
  <div class="et">«{_esc(r["etiqueta"])}» <span class="nid">{_esc(r["id"])}</span>{badge} <span class="fr">{_esc(r["frases"])}</span></div>
  <div class="track"><div class="fill" style="width:{100 * r["n"] / n_max:.1f}%;background:var(--c{r["i"] + 1})"></div></div>
  <div class="post"><span class="mid"></span><span class="ap" style="width:{min(50, 50 * r["apoya"]):.0f}%"></span><span class="re" style="width:{min(50, 50 * r["rechaza"]):.0f}%"></span></div>
  <div class="num">{_num(r["n"])}</div>
</div>''')
    kpis = [(_num(d["total"]), "comentarios"), (d["dias"], "días"), (d["n_narr"], "narrativas"),
            (d["n_emerg"], "emergentes"), (f'{d["n_grupos"]} <small>· {_num(d["n_coord"])} cuentas</small>', "grupos coord.")]
    kpis_h = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in kpis)
    costo = d["costo"]
    usd_mil = costo.get("usd_computo_por_mil")
    if usd_mil is not None:
        usd_mil = usd_mil + (costo.get("usd_por_mil_comentarios") or 0)
    titulo5 = (f"Procesar {_num(d['total'])} comentarios llevó {_num(costo.get('segundos', 0), 1)} s"
               + (f" y USD {_usd(usd_mil)} por mil" if usd_mil is not None else ""))
    alm = costo.get("almacenamiento_mb") or {}
    kpis5 = [f'{_num(costo.get("segundos", 0), 1)} s|tiempo total',
             f'{_num(costo.get("ram_pico_gb") or 0, 2) if costo.get("ram_pico_gb") is not None else "-"} GB|RAM pico',
             (f'{_num(alm.get("duckdb", 0), 1)} / {_num(alm.get("parquet", 0), 1)} / '
              f'{_num(alm.get("entrada", 0), 1)} MB|duckdb / parquet / entrada'),
             f'USD {_usd(costo.get("usd_ia"))}|IA ({_esc(costo.get("modo_llm", ""))})']
    proy = (costo.get("proyeccion_1M_por_dia") or {})
    proy_txt = ""
    if proy and proy.get("horas") is not None:
        proy_txt = ("estimación (extrapolación lineal): "
                    + f"{float(proy['horas']):.2g}".replace(".", ",") + " h para 1M/día")
    notas5 = "; ".join(x for x in [costo.get("notas", ""), proy_txt,
                                   "corrida en caliente: etapas reutilizadas, no representa el "
                                   "costo en frío" if d["caliente"] else ""] if x)
    grupos_btn = "".join(f'<button class="gbtn" data-g="{_esc(t["id"])}">{_esc(t["id"])}</button>'
                         for t in d["tarjetas"])
    tabs = "".join(f'<button class="tab" data-p="{i + 1}">[{i + 1} {n}]</button>' for i, n in enumerate(PASOS))
    secciones = "".join(
        f'<section class="paso" id="p{i + 1}"{"" if i == 0 else " hidden"}><h2 class="pt"></h2><div class="pc"></div></section>'
        for i in range(len(PASOS)))
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QualyData · {_esc(d['titulo_set'])} · tablero</title>
<style>
:root {{ --bg:#0b0d12; --sup:#141824; --sup2:#1c2233; --txt:#eef0f6; --sec:#aab0c0; --mut:#767e92;
  --borde:#2a3040; --apoya:#4C9BE8; --rechaza:#F28C28; --otras:{GRIS};
  {''.join(f"--c{i + 1}:{h};" for i, h in enumerate(PALETA))} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--txt); font:16px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;
  display:flex; flex-direction:column; height:100vh; }}
header {{ display:flex; justify-content:space-between; align-items:center; padding:10px 20px 6px; flex:none; }}
h1 {{ font-size:22px; margin:0; }}
#todo {{ background:var(--sup2); color:var(--txt); border:1px solid var(--borde); border-radius:8px;
  padding:6px 14px; font-size:15px; cursor:pointer; }}
nav {{ display:flex; gap:8px; padding:4px 20px 10px; flex:none; }}
.tab, .gbtn {{ background:var(--sup); color:var(--sec); border:1px solid var(--borde); border-radius:8px;
  padding:6px 14px; font-size:15px; cursor:pointer; }}
.tab.on, .gbtn.on {{ color:var(--txt); border-color:var(--apoya); background:var(--sup2); }}
main {{ flex:1; min-height:0; }}
.paso {{ height:100%; padding:0 20px; overflow:hidden; display:flex; flex-direction:column; }}
.paso[hidden] {{ display:none; }}
.pt {{ font-size:19px; margin:2px 0 10px; font-weight:600; flex:none; }}
.pc {{ flex:1; min-height:0; }}
.kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:14px; }}
.kpi {{ background:var(--sup); border-radius:10px; padding:10px 14px; }}
.kpi .v {{ font-size:34px; font-weight:700; font-variant-numeric:tabular-nums; }}
.kpi .v small {{ font-size:16px; color:var(--sec); }}
.kpi .k {{ color:var(--sec); font-size:13px; }}
#p1 .pc {{ display:flex; flex-direction:column; }}
.barras {{ flex:1; min-height:0; overflow:hidden; }}
.barr, .barr-head {{ display:grid; grid-template-columns:14px 1fr 110px 150px 80px; gap:10px;
  align-items:center; padding:3px 10px; border-radius:8px; }}
.barr {{ cursor:pointer; height:30px; }}
.barr:hover {{ background:var(--sup); }}
.barr.dim {{ opacity:.25; }}
.barr.oculta {{ display:none; }}
.barr-head {{ color:var(--mut); font-size:12px; height:22px; }}
.barr-head .ph {{ grid-column:4; text-align:center; }}
.barr-head .nh {{ grid-column:5; text-align:right; }}
.barr.mas {{ color:var(--mut); font-size:13px; font-style:italic; cursor:default;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.sw {{ width:12px; height:12px; border-radius:3px; border:1px solid var(--mut); }}
.et {{ font-size:15px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.nid {{ color:var(--mut); font-size:12px; }}
.fr {{ color:var(--mut); font-size:12px; }}
.track {{ height:7px; background:var(--sup2); border-radius:4px; overflow:hidden; }}
.fill {{ height:100%; border-radius:4px; }}
.post {{ height:12px; border-radius:6px; background:var(--sup2); position:relative; }}
.post .ap {{ position:absolute; right:50%; top:0; bottom:0; background:var(--apoya); border-radius:6px 0 0 6px; }}
.post .re {{ position:absolute; left:50%; top:0; bottom:0; background:var(--rechaza); border-radius:0 6px 6px 0; }}
.post .mid {{ position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--borde); }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; color:var(--sec); }}
.badge {{ background:#3a2a12; color:#f5c46a; border:1px solid #6a4d1e; border-radius:6px;
  padding:1px 7px; font-size:11px; font-weight:600; }}
.pie {{ color:var(--mut); font-size:13px; margin-top:8px; }}
.graf {{ width:100%; height:calc(100% - 70px); }}
.cols {{ display:grid; grid-template-columns:1fr 380px; gap:14px; flex:1; min-height:0; }}
.cards {{ overflow-y:auto; }}
.card {{ background:var(--sup); border:1px solid var(--borde); border-radius:10px; padding:10px 14px;
  margin-bottom:10px; cursor:pointer; }}
.card:hover {{ border-color:var(--apoya); }}
.card.dim {{ opacity:.3; }}
.nums {{ display:flex; gap:22px; margin:8px 0; }}
.nums .v {{ font-size:30px; font-weight:700; font-variant-numeric:tabular-nums; }}
.nums .k {{ color:var(--sec); font-size:12px; }}
.mini {{ width:100%; height:150px; }} .mini2 {{ width:100%; height:170px; }}
#tt {{ position:fixed; display:none; max-width:420px; background:var(--sup2); border:1px solid var(--borde);
  border-radius:8px; padding:8px 12px; font-size:13px; color:var(--sec); pointer-events:none; z-index:9; }}
#tt li {{ margin:3px 0; font-style:italic; }}
@media (max-width:1100px) {{ .cols {{ grid-template-columns:1fr; }} .kpis {{ grid-template-columns:repeat(3,1fr); }} }}
</style>{plotly_js()}</head>
<body>
<header><h1>QualyData · {_esc(d['titulo_set'])}</h1><button id="todo">[ver todo]</button></header>
<nav>{tabs}</nav>
<main>
<section class="paso" id="p1"><h2 class="pt">{_esc(_frase1(d))}</h2><div class="pc">
<div class="kpis">{kpis_h}</div>
<div class="barras" id="barras1"><div class="barr-head"><span></span><span></span><span></span><span class="ph">◀ apoya · rechaza ▶</span><span class="nh">comentarios</span></div>{''.join(filas_p1)}<div class="barr mas" id="masrow" hidden></div></div>
<p class="pie">Los bandos se separan bien; decidir cuál apoya falla en algunas narrativas (macro-F1 0,65). · ruido: {100 * d['ruido']:.0f}% de los comentarios.</p>
</div></section>
<section class="paso" id="p2" hidden><h2 class="pt" id="t2"></h2><div class="pc"><div id="g2" class="graf"></div>
<p class="pie">La fecha de origen es una estimación.</p></div></section>
<section class="paso" id="p3" hidden><h2 class="pt">{_esc(_plural(d['n_grupos'], 'enjambre de cuentas actúa', 'enjambres de cuentas actúan'))} juntos mucho más que lo esperable por azar</h2>
<div class="pc cols"><div id="g3" class="graf" style="height:100%"></div><div class="cards" id="cards3"></div></div>
<p class="pie">Coordinación anómala detectada: una alerta para un analista, no prueba de manipulación. × = cohesión observada / cohesión con cuentas al azar.</p></section>
<section class="paso" id="p4" hidden><h2 class="pt" id="t4"></h2><div class="pc">
<div class="gbts">{grupos_btn}</div><div id="p4body"></div>
<p class="pie">Coordinación anómala detectada: una alerta para un analista, no prueba de manipulación.</p></div></section>
<section class="paso" id="p5" hidden><h2 class="pt">{_esc(titulo5)}</h2><div class="pc">
<div class="kpis">{''.join(f'<div class="kpi"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in (x.split("|") for x in kpis5))}</div>
<div id="g5" class="graf" style="height:280px"></div>
<p class="pie">{_esc(notas5)}</p></div></section>
</main>
<div id="tt"></div>
<script>
const D = {payload};
const $ = s => document.querySelector(s);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const CFG = {{ displaylogo:false, responsive:true, displayModeBar:false }};
const base = () => ({{ paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{ family:'system-ui, sans-serif', size:13, color:css('--sec') }},
  margin:{{ l:52, r:16, t:36, b:42 }},
  xaxis:{{ gridcolor:css('--borde'), zeroline:false }},
  yaxis:{{ gridcolor:css('--borde'), zeroline:false, rangemode:'tozero' }} }});
const colorDe = i => i >= 0 ? css('--c' + (i % {len(PALETA)} + 1)) : css('--otras');
const fmtN = (x, dec = 0) => Number(x).toLocaleString('es-AR', {{ maximumFractionDigits:dec }});
let sel = {{ tipo:null, id:null }};
const filaDe = id => D.filas.find(f => f.id === id);

function compactarP1() {{
  const cont = $('#barras1'), mas = $('#masrow');
  if (!cont) return;
  const rows = [...cont.querySelectorAll('.barr:not(.mas)')];
  rows.forEach(r => r.classList.remove('oculta'));
  mas.hidden = true;
  const ocultas = [];
  // primero oculta las no emergentes desde el fondo; emergentes siempre visibles si entran
  for (let i = rows.length - 1; i >= 0 && cont.scrollHeight > cont.clientHeight; i--)
    if (rows[i].dataset.emerg !== '1') {{ rows[i].classList.add('oculta'); ocultas.push(rows[i]); }}
  for (let i = rows.length - 1; i >= 0 && cont.scrollHeight > cont.clientHeight; i--)
    if (!rows[i].classList.contains('oculta')) {{ rows[i].classList.add('oculta'); ocultas.push(rows[i]); }}
  if (ocultas.length) {{
    mas.hidden = false;
    const ncom = ocultas.reduce((a, r) => a + Number(r.dataset.n), 0);
    mas.textContent = '+' + ocultas.length + (ocultas.length === 1 ? ' narrativa más chica' : ' narrativas más chicas') +
      ' (' + fmtN(ncom) + ' comentario' + (ncom === 1 ? '' : 's') + ')';
  }}
}}

function irPaso(n, sub) {{
  location.hash = sub ? n + '/' + sub : String(n);
}}
function aplicarHash() {{
  const m = (location.hash || '#1').slice(1).split('/');
  const n = Math.min(5, Math.max(1, parseInt(m[0]) || 1));
  document.querySelectorAll('.paso').forEach((s, i) => s.hidden = i !== n - 1);
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('on', i === n - 1));
  if (m[1]) {{ sel = m[1].startsWith('n') ? {{ tipo:'nar', id:m[1] }} : {{ tipo:'grupo', id:m[1] }}; }}
  dibujar(n);
  pintarSel();
  compactarP1();
}}
function pintarSel() {{
  document.querySelectorAll('[data-nar]').forEach(e =>
    e.classList.toggle('dim', sel.tipo === 'nar' && e.dataset.nar !== sel.id));
}}
function verTodo() {{ sel = {{ tipo:null, id:null }}; pintarSel(); dibujar(pasoActual()); }}
const pasoActual = () => 1 + [...document.querySelectorAll('.paso')].findIndex(s => !s.hidden);

document.querySelectorAll('.tab').forEach(t => t.onclick = () => irPaso(+t.dataset.p));
document.querySelectorAll('.gbtn').forEach(b => b.onclick = () => irPaso(4, b.dataset.g));
$('#todo').onclick = verTodo;
document.querySelectorAll('.barr:not(.mas)').forEach(b => {{
  b.onclick = () => {{ sel = sel.id === b.dataset.nar ? {{ tipo:null, id:null }} : {{ tipo:'nar', id:b.dataset.nar }}; pintarSel(); dibujar(pasoActual()); }};
  b.onmouseenter = e => {{
    const tt = $('#tt');
    tt.innerHTML = '<div style="color:var(--mut)">' + (b.dataset.pos || '') + '</div><ul>' +
      JSON.parse(b.dataset.ej).map(x => '<li>«' + x + '»</li>').join('') + '</ul>';
    tt.style.display = 'block';
  }};
  b.onmousemove = e => {{ const tt = $('#tt'); tt.style.left = (e.clientX + 14) + 'px'; tt.style.top = (e.clientY + 14) + 'px'; }};
  b.onmouseleave = () => $('#tt').style.display = 'none';
}});
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') return verTodo();
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {{
    const n = pasoActual() + (e.key === 'ArrowRight' ? 1 : -1);
    if (n >= 1 && n <= 5) irPaso(n);
  }}
  if (e.key >= '1' && e.key <= '5') irPaso(+e.key);
}});
window.addEventListener('hashchange', aplicarHash);
window.addEventListener('resize', compactarP1);

function dibujar(n) {{
  if (n === 2) paso2();
  if (n === 3) paso3();
  if (n === 4) paso4(sel.tipo === 'grupo' ? sel.id :
    ((D.tarjetas.reduce((a, b) => (b.nx_v > ((a || {{}}).nx_v || 0) ? b : a), null) || {{}}).id));
  if (n === 5) paso5();
}}

function paso2() {{
  const foco = sel.tipo === 'nar' && D.serie_dia[sel.id] ? sel.id : D.emerg[0];
  const maxDia = lab => Math.max(0, ...(D.serie_dia[lab] || [0]));
  if (foco) {{
    const f = filaDe(foco);
    $('#t2').textContent = '«' + f.etiqueta + '» ' + (f.emergente ? 'apareció' : 'circula desde') +
      ' el ' + f.origen_fecha + ' y llegó a ' + fmtN(maxDia(foco)) + ' comentarios por día';
  }} else $('#t2').textContent = 'Sin narrativas emergentes en este set';
  const shown = sel.tipo === 'nar' && D.serie_dia[sel.id] ? [sel.id] : D.emerg;
  const enSerie = new Set(shown);
  const resto = D.x.map((_, k) => Object.entries(D.serie_dia).reduce((a, [lab, y]) =>
    a + (enSerie.has(lab) ? 0 : y[k]), 0));
  const tr = shown.map(lab => ({{ x:D.x, y:D.serie_dia[lab], name:lab + ' · ' + filaDe(lab).etiqueta.slice(0, 40),
    type:'scatter', mode:'lines', stackgroup:'uno', line:{{ width:1, color:css('--bg') }},
    fillcolor:colorDe(filaDe(lab).i) }}));
  tr.push({{ x:D.x, y:resto, name:'resto', type:'scatter', mode:'lines', stackgroup:'uno',
    line:{{ width:1, color:css('--bg') }}, fillcolor:css('--otras'), opacity:.55 }});
  const L = base();
  L.yaxis.title = {{ text:'comentarios por día' }};
  L.xaxis.rangeslider = {{ thickness:.08 }};
  L.showlegend = true; L.legend = {{ x:1.01, y:1, xanchor:'left', font:{{ size:11 }} }};
  L.margin.r = 250;
  L.shapes = shown.filter(l => D.origenes[l]).map(lab => ({{ type:'line',
    x0:D.origenes[lab].slice(0, 10), x1:D.origenes[lab].slice(0, 10), yref:'paper', y0:0, y1:1,
    line:{{ dash:'dash', width:2, color:colorDe(filaDe(lab).i) }} }}));
  L.annotations = shown.filter(l => D.origenes[l]).map((lab, k) => ({{
    x:D.origenes[lab].slice(0, 10), yref:'paper', y:1.0 - 0.09 * (k % 5), yanchor:'bottom',
    text:'origen ' + lab, showarrow:false, bgcolor:css('--bg'), borderpad:2,
    font:{{ size:11, color:css('--sec') }} }}));
  Plotly.react('g2', tr, L, CFG);
}}

function paso3() {{
  const g = D.grafo;
  const ei = {{ x:[], y:[] }};
  // solo aristas intra-cluster: dentro de un grupo o entre normales; las que cruzan se omiten
  for (const [a, b] of g.aristas) {{
    if ((g.nodos[a].g || '') !== (g.nodos[b].g || '')) continue;
    ei.x.push(g.nodos[a].x, g.nodos[b].x, null); ei.y.push(g.nodos[a].y, g.nodos[b].y, null);
  }}
  const COLG = ['#e8e8e8', '#f5d76e', '#4dd8e8', '#f0a0c8', '#a8d878', '#c898f0'];
  const SYMS = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'star'];
  const colG = {{}}, symG = {{}};
  D.tarjetas.forEach((t, i) => {{ colG[t.id] = COLG[i % COLG.length]; symG[t.id] = SYMS[i % SYMS.length]; }});
  const porGrupo = {{}};
  g.nodos.forEach(nd => (porGrupo[nd.g || 'normal'] = porGrupo[nd.g || 'normal'] || []).push(nd));
  const tr = [{{ x:ei.x, y:ei.y, mode:'lines', type:'scatter', hoverinfo:'skip',
    line:{{ width:.5, color:'rgba(140,150,170,.28)' }}, showlegend:false }}];
  const ordenLey = ['normal', ...D.tarjetas.map(t => t.id)];
  for (const gid of ordenLey) {{
    const ns = porGrupo[gid];
    if (!ns) continue;
    tr.push({{ x:ns.map(n => n.x), y:ns.map(n => n.y), mode:'markers', type:'scatter',
      name:gid === 'normal' ? 'muestra de cuentas no coordinadas' : gid,
      text:ns.map(n => n.c + (n.g ? ' · ' + n.g : '') + (n.tip ? '<br>«' + n.tip.replace(/</g,'&lt;') + '»' : '')),
      hovertemplate:'%{{text}}<extra></extra>',
      marker:{{ size:gid === 'normal' ? 4 : 9, color:gid === 'normal' ? css('--otras') : colG[gid],
        symbol:gid === 'normal' ? 'circle' : symG[gid],
        opacity:gid === 'normal' ? .4 : .95,
        line:gid === 'normal' ? undefined : {{ width:1, color:css('--bg') }} }} }});
  }}
  const L = base();
  L.xaxis = {{ visible:false }}; L.yaxis = {{ visible:false }};
  L.showlegend = true; L.legend = {{ orientation:'h', y:-0.05 }};
  L.margin = {{ l:8, r:8, t:8, b:36 }};
  Plotly.react('g3', tr, L, CFG);
  const cont = $('#cards3');
  const et40 = e => e.etiqueta.length > 40 ? e.etiqueta.slice(0, 40) + '…' : e.etiqueta;
  cont.innerHTML = D.tarjetas.map(t =>
    '<div class="card" data-g="' + t.id + '"><b>' + t.id + '</b> · ' + t.n + ' cuentas · <b>' + t.nx + '</b>' +
    (t.empuja.length ? '<div class="fr">' + fmtN(t.empuja[0].pct, 1) + ' % en «' + et40(t.empuja[0]) + '»</div>' : '') +
    (t.empuja.length > 1 ? '<div class="fr">' +
      t.empuja.slice(1).map(e => e.id + ' ' + fmtN(e.pct, 1) + ' %').join(' · ') + '</div>' : '') +
    '<div class="track">' + t.empuja.map(e => '<div class="fill" style="width:' + Math.min(100, e.pct) +
      '%;background:' + colorDe(e.i) + '"></div>').join('') + '</div></div>').join('');
  cont.querySelectorAll('.card').forEach(c => c.onclick = () => {{ sel = {{ tipo:'grupo', id:c.dataset.g }}; irPaso(4, c.dataset.g); }});
}}

function paso4(gid) {{
  const t = D.tarjetas.find(x => x.id === gid) || D.tarjetas[0];
  if (!t) {{ $('#p4body').innerHTML = '<p class="pie">No se detectaron operaciones coordinadas.</p>'; return; }}
  document.querySelectorAll('.gbtn').forEach(b => b.classList.toggle('on', b.dataset.g === t.id));
  $('#t4').textContent = t.top_pct >= 50
    ? 'El grupo ' + t.id + ' publica el ' + fmtN(t.top_pct, 1) + ' % de sus comentarios en «' + t.top_et + '»'
    : 'Evidencia del grupo ' + t.id;
  $('#p4body').innerHTML =
    '<div class="nums"><div><div class="v">' + t.nx + '</div><div class="k">co-actúan vs azar</div></div>' +
    '<div><div class="v">' + fmtN(t.posts) + '</div><div class="k">publicaciones compartidas</div></div>' +
    '<div><div class="v">' + fmtN(t.respuestas) + '</div><div class="k">respuestas mutuas</div></div>' +
    '<div><div class="v">' + fmtN(t.n_tags) + '</div><div class="k">hashtags raros' +
    (t.tags.length ? ': ' + t.tags.slice(0, 3).join(' ') : '') + '</div></div></div>' +
    '<div class="cols" style="grid-template-columns:1fr 1fr;height:auto">' +
    '<div><div class="fr">% de comentarios por hora: ' + t.id + ' vs no coordinadas</div><div id="h4" class="mini"></div></div>' +
    '<div><div class="fr">«' + (t.post.titular || '').replace(/</g,'&lt;') + '» — un post, minuto a minuto</div><div id="g4" class="mini2"></div></div></div>' +
    '<div class="fr">comentarios del grupo:</div><ul style="color:var(--sec);font-size:14px;margin-top:2px">' +
    t.ej.slice(0, 3).map(x => '<li>«' + x.replace(/</g,'&lt;') + '»</li>').join('') + '</ul>';
  const Lh = base(); Lh.margin = {{ l:70, r:60, t:4, b:30 }};
  Lh.xaxis = {{ tickmode:'array', tickvals:[0,3,6,9,12,15,18,21], tickfont:{{ size:9 }}, gridcolor:css('--borde'),
    title:{{ text:'hora local (−03:00)', font:{{ size:10 }} }} }};
  Plotly.react('h4', [{{ z:[t.horas, D.horas_ng], y:[t.id, 'resto'], x:[...Array(24).keys()],
    type:'heatmap', showscale:true, colorscale:'Viridis',
    colorbar:{{ thickness:8, tickfont:{{ size:9 }}, title:{{ text:'% com.', font:{{ size:9 }} }} }} }}], Lh, CFG);
  const L4 = base(); L4.margin = {{ l:10, r:10, t:4, b:30 }};
  L4.xaxis = {{ title:{{ text:'minutos desde el primero', font:{{ size:10 }} }},
    gridcolor:css('--borde'), zeroline:false }};
  L4.yaxis = {{ type:'category', showticklabels:false, showgrid:false, ticks:'' }};
  Plotly.react('g4', [{{ x:t.post.pts.map(p => p.m), y:t.post.pts.map(p => p.c), mode:'markers',
    type:'scatter', marker:{{ size:7, color:css('--apoya') }},
    hovertemplate:'%{{y}} · +%{{x}} min<extra></extra>' }}], L4, CFG);
}}

function paso5() {{
  const L = base(); L.margin.l = 120;
  L.xaxis.title = {{ text:'segundos' }}; L.yaxis.autorange = 'reversed';
  L.showlegend = true; L.legend = {{ orientation:'h', y:-0.15 }};
  Plotly.react('g5', [{{ x:Object.values(D.etapas), y:Object.keys(D.etapas), type:'bar',
    orientation:'h', name:'segundos por etapa', marker:{{ color:css('--apoya') }},
    hovertemplate:'%{{y}}: %{{x:.1f}} s<extra></extra>' }}], L, CFG);
}}

aplicarHash();
</script></body></html>"""


def correr(ctx) -> None:
    try:
        run(ctx)
    except Exception as e:  # noqa: BLE001 — la demo nunca rompe el pipeline
        log.warning("e11_demo falló: %s", e)


def run(ctx) -> None:
    import time

    t = time.perf_counter()
    costo = dict(ctx.costo.a_dict())
    from qd import costo as qcosto
    from qd.config import obtener

    costo["almacenamiento_mb"] = qcosto.medir_almacenamiento(ctx.datos, ctx.work)
    d = recolectar(ctx.db, ctx.cfg, costo, ctx.datos.name)
    destino = Path(obtener(ctx.cfg, "reporte.dir", "reportes")) / f"{ctx.datos.name}_demo.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(render(d), encoding="utf-8")
    log.info("e11: tablero -> %s (%.1fs)", destino, time.perf_counter() - t)


class _DBReadOnly:
    """Mínima interfaz db.leer/db.existe sobre un DuckDB abierto en solo-lectura."""

    def __init__(self, ruta: Path):
        import duckdb

        self.con = duckdb.connect(str(ruta), read_only=True)

    def leer(self, nombre):
        return self.con.execute(f'SELECT * FROM "{nombre}"').df()

    def existe(self, nombre):
        return self.con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                                [nombre]).fetchone()[0] > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--costo", type=Path, required=True)
    ap.add_argument("--set", required=True, help="nombre del set (practica, final, …)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    a = ap.parse_args(argv)
    try:
        from qd import config as qcfg

        cfg = qcfg.cargar(a.config) if a.config.exists() else {}
        costo = json.loads(a.costo.read_text(encoding="utf-8"))
        d = recolectar(_DBReadOnly(a.db), cfg, costo, a.set)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(render(d), encoding="utf-8")
        print(f"e11: tablero -> {a.out}")
        return 0
    except Exception as e:  # noqa: BLE001
        log.warning("e11_demo falló: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
