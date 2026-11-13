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
import base64
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
# Ámbar de acento de UI (etiquetas/emergentes); ningún color de PALETA es ámbar.
AMBAR = "#E3B341"
KICKERS = ["HALLAZGO PRINCIPAL", "DETECCIÓN DE EMERGENTES", "GRUPOS COORDINADOS",
           "EVIDENCIA", "COSTO DE LA CORRIDA"]
_FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def _fuentes() -> str:
    """@font-face con woff2 en base64 (subconjunto latino, OFL en qd/assets/fonts/).
    Si falta un archivo se omite la cara y quedan los respaldos Georgia/monospace."""
    caras = [("Newsreader", "normal", "100 900", "newsreader.woff2"),
             ("Newsreader", "italic", "100 900", "newsreader-italic.woff2"),
             ("JetBrains Mono", "normal", "100 800", "jetbrains-mono.woff2")]
    sal = []
    for fam, est, peso, arch in caras:
        p = _FONTS / arch
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        sal.append(f"@font-face{{font-family:'{fam}';font-style:{est};font-weight:{peso};"
                   f"font-display:swap;src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "".join(sal)


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
    """es-AR: 2 decimales si >= 0,01; si no, 2 cifras significativas sin ceros
    de relleno ('0,0005', '0,000031'); 0 exacto -> '0'."""
    x = float(x or 0)
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return _num(x, 2)
    sig = float(f"{x:.2g}")
    return f"{sig:.10f}".rstrip("0").rstrip(".").replace(".", ",")


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
        "grafo": grafo, "costo": costo, "delta_min": delta / 60,
        "caliente": "e01_ingesta" not in (costo.get("desglose_segundos") or {}),
        "etapas": costo.get("desglose_segundos") or {},
    }


def _frase1(d: dict) -> tuple:
    """(titular, sub): la segunda oración (grupo que concentra >= 50 %) va aparte,
    en itálica ámbar debajo del titular."""
    s = (f"En {d['dias']} días circularon {_plural(d['n_narr'], 'narrativa', 'narrativas')}; "
         f"{d['n_emerg']} aparecieron de golpe y {_plural(d['n_grupos'], 'grupo coordinado', 'grupos coordinados')} "
         f"({_plural(d['n_coord'], 'cuenta', 'cuentas')}) actúan en bloque.")
    sub = ""
    cand = [t for t in d["tarjetas"] if t["top_pct"] >= 50 and t["top_et"]]
    if cand:
        t = max(cand, key=lambda t: (t["nx_v"], t["top_pct"]))
        sub = f'El grupo {t["id"]} concentra el {t["top_pct"]:.0f} % de su actividad en «{_esc(t["top_et"])}».'
    return s, sub


def render(d: dict) -> str:
    from qd.etapas.e10_reporte import plotly_js

    payload = json.dumps(d, ensure_ascii=False, default=str)
    filas_p1 = []
    for r in d["filas"]:
        badge = ' <span class="badge">EMERGENTE</span>' if r["emergente"] else ""
        pos_txt = (f"postura: {100 * r['apoya']:.0f} % apoya · {100 * r['rechaza']:.0f} % rechaza · "
                   f"{100 * (1 - r['apoya'] - r['rechaza']):.0f} % sin bando")
        filas_p1.append(f'''<div class="barr{" emg" if r["emergente"] else ""}" data-nar="{_esc(r["id"])}" data-emerg="{1 if r["emergente"] else 0}"
  data-n="{r["n"]}" data-ej="{_esc(json.dumps(r["ej"][:2]))}" data-pos="{_esc(pos_txt)}">
  <span class="sw" style="background:{'var(--c' + str(r["i"] + 1) + ')' if r["i"] >= 0 else 'var(--otras)'}"></span>
  <div class="et"><div class="e1">«{_esc(r["etiqueta"])}»{badge}</div>
    <div class="e2">{_esc(r["id"])} · {_esc(r["frases"])}</div></div>
  <div class="post"><span class="mid"></span><span class="ap" style="width:{min(50, 50 * r["apoya"]):.0f}%"></span><span class="re" style="width:{min(50, 50 * r["rechaza"]):.0f}%"></span></div>
  <div class="num">{_num(r["n"])}</div>
</div>''')
    kpis = [(_num(d["total"]), "comentarios", ""), (d["dias"], "días", ""),
            (_num(d["n_narr"]), "narrativas", ""), (_num(d["n_emerg"]), "emergentes", " amb"),
            (_num(d["n_grupos"]), f"grupos coord. · {_num(d['n_coord'])} cuentas", " az")]
    kpis_h = "".join(f'<div class="kpi{cls}"><div class="v">{v}</div><div class="k">{k}</div></div>'
                     for v, k, cls in kpis)
    costo = d["costo"]
    usd_mil = costo.get("usd_computo_por_mil")
    if usd_mil is not None:
        usd_mil = usd_mil + (costo.get("usd_por_mil_comentarios") or 0)
    titulo5 = (f"Procesar {_num(d['total'])} comentarios llevó {_num(costo.get('segundos', 0), 1)} s"
               + (f" y USD {_usd(usd_mil)} por mil" if usd_mil is not None else ""))
    alm = costo.get("almacenamiento_mb") or {}
    tot_mb = alm.get("duckdb", 0) + alm.get("parquet", 0) + alm.get("entrada", 0)
    kpis5 = [f'{_num(costo.get("segundos", 0), 1)} s|tiempo total',
             f'{_num(costo.get("ram_pico_gb") or 0, 2) if costo.get("ram_pico_gb") is not None else "-"} GB|RAM pico',
             ((f'{_num(tot_mb, 1)} MB|duckdb {_num(alm.get("duckdb", 0), 1)} · '
               f'parquet {_num(alm.get("parquet", 0), 1)} · entrada {_num(alm.get("entrada", 0), 1)} MB')
              if alm else '-|almacenamiento'),
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
    tit1, sub1 = _frase1(d)
    sub1_h = f'<p class="sub">{_esc(sub1)}</p>' if sub1 else ""
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QualyData · {_esc(d['titulo_set'])} · tablero</title>
<style>{_fuentes()}
:root {{ --bg:#0b0d12; --sup:#12161f; --sup2:#1a2030; --txt:#eef0f6; --sec:#aab0c0; --mut:#828aa0;
  --borde:#242b3a; --apoya:#4C9BE8; --rechaza:#F28C28; --otras:{GRIS};
  --amb:{AMBAR}; --amb-bg:rgba(227,179,65,.08); --amb-bo:rgba(227,179,65,.45);
  --az-bg:rgba(76,155,232,.08); --az-bo:rgba(76,155,232,.45);
  --serif:'Newsreader',Georgia,serif; --mono:'JetBrains Mono',monospace;
  {''.join(f"--c{i + 1}:{h};" for i, h in enumerate(PALETA))} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--txt); font:16px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
  display:flex; flex-direction:column; height:100vh; }}
header {{ display:flex; align-items:center; gap:20px; padding:10px 22px; flex:none;
  border-bottom:1px solid var(--borde); background:#0e1119; }}
.brand {{ display:flex; align-items:center; gap:10px; }}
.dot {{ width:11px; height:11px; border-radius:50%; background:var(--amb);
  box-shadow:0 0 0 4px rgba(227,179,65,.16); }}
.bname {{ font:700 20px var(--serif); color:var(--txt); }}
.bset {{ font:600 14px var(--mono); color:var(--mut); text-transform:uppercase; letter-spacing:.08em; }}
nav {{ margin-left:auto; display:flex; gap:4px; background:#090b10;
  border:1px solid var(--borde); border-radius:10px; padding:4px; }}
.tab {{ background:transparent; border:0; border-radius:7px; color:var(--mut);
  font:500 14px var(--mono); padding:6px 12px; cursor:pointer; }}
.tab.on {{ background:var(--sup2); color:var(--txt); box-shadow:inset 0 0 0 1px #3b4763; }}
#todo {{ font:500 14px var(--mono); background:var(--sup); color:var(--sec);
  border:1px solid var(--borde); border-radius:8px; padding:7px 13px; cursor:pointer; }}
#todo:hover {{ color:var(--txt); }}
main {{ flex:1; min-height:0; }}
.paso {{ height:100%; padding:0 22px; overflow:hidden; display:flex; flex-direction:column; }}
.paso[hidden] {{ display:none; }}
.kick {{ align-self:flex-start; font:700 14px var(--mono); letter-spacing:.12em;
  text-transform:uppercase; color:var(--amb); background:var(--amb-bg);
  border:1px solid var(--amb-bo); border-radius:6px; padding:3px 10px; margin-top:8px; }}
.pt {{ font:600 28px/1.22 var(--serif); color:var(--txt); margin:8px 0 2px; flex:none; }}
.sub {{ font:italic 18px var(--serif); color:var(--amb); margin:0 0 4px; flex:none; }}
.pc {{ flex:1; min-height:0; display:flex; flex-direction:column; }}
.kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:4px 0 10px; flex:none; }}
.kpi {{ background:var(--sup); border:1px solid var(--borde); border-radius:10px; padding:8px 14px; }}
.kpi .v {{ font:700 44px/1.1 var(--mono); color:var(--txt); }}
.kpi .v small {{ font-size:16px; color:var(--sec); }}
.kpi .k {{ font:600 14px var(--mono); text-transform:uppercase; letter-spacing:.06em; color:var(--mut); }}
.kpi.amb {{ background:var(--amb-bg); border-color:var(--amb-bo); }}
.kpi.amb .v, .kpi.amb .k {{ color:var(--amb); }}
.kpi.az {{ background:var(--az-bg); border-color:var(--az-bo); }}
.kpi.az .v, .kpi.az .k {{ color:var(--apoya); }}
.barras {{ flex:1; min-height:0; overflow:hidden; }}
.barr, .barr-head {{ display:grid; grid-template-columns:16px 1fr 170px 90px; gap:12px;
  align-items:center; padding:4px 10px; border-radius:8px; }}
.barr {{ cursor:pointer; border:1px solid transparent; }}
.barr:hover {{ background:var(--sup); border-color:var(--borde); }}
.barr.emg {{ background:var(--amb-bg); border-color:var(--amb-bo); }}
.barr.dim {{ opacity:.25; }}
.barr.oculta {{ display:none; }}
.barr-head {{ font:600 14px var(--mono); color:var(--mut); text-transform:uppercase;
  letter-spacing:.06em; border-bottom:1px solid var(--borde); border-radius:0; padding-bottom:6px; }}
.barr-head .h-lab {{ grid-column:1/3; }}
.barr-head .ph {{ grid-column:3; display:flex; justify-content:center; gap:14px; }}
.barr-head .nh {{ grid-column:4; text-align:right; }}
.sq {{ display:inline-block; width:11px; height:11px; border-radius:3px;
  margin-right:6px; vertical-align:-1px; }}
.barr.mas {{ color:var(--mut); font:italic 14px var(--mono); cursor:default;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.sw {{ width:12px; height:12px; border-radius:50%; border:1px solid var(--mut); }}
.et {{ min-width:0; }}
.e1 {{ font:17px/1.25 var(--serif); color:var(--txt); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }}
.e2 {{ font:14px var(--mono); color:var(--mut); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }}
.badge {{ font:700 14px var(--mono); background:rgba(227,179,65,.16); color:var(--amb);
  border:1px solid var(--amb-bo); border-radius:6px; padding:0 7px; margin-left:8px; vertical-align:1px; }}
.post {{ height:10px; border-radius:5px; background:var(--sup2); position:relative; }}
.post .ap {{ position:absolute; right:50%; top:0; bottom:0; background:var(--apoya); border-radius:5px 0 0 5px; }}
.post .re {{ position:absolute; left:50%; top:0; bottom:0; background:var(--rechaza); border-radius:0 5px 5px 0; }}
.post .mid {{ position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--borde); }}
.num {{ font:600 16px var(--mono); text-align:right; color:var(--txt); }}
.pie {{ font:14px var(--mono); color:var(--mut); margin:6px 0 8px; flex:none; }}
.graf {{ width:100%; min-height:0; }}
.cols {{ display:grid; grid-template-columns:1fr 380px; gap:14px; flex:1; min-height:0; }}
.gwrap {{ display:flex; flex-direction:column; min-height:0; }}
.side {{ display:flex; flex-direction:column; gap:10px; overflow:hidden; }}
.ecard {{ background:var(--sup); border:1px solid var(--borde); border-radius:10px;
  padding:8px 12px; cursor:pointer; }}
.ecard.on {{ border-color:var(--amb-bo); }}
.ecard .l {{ font:16px/1.25 var(--serif); color:var(--txt); }}
.ecard .l .sw {{ vertical-align:0; margin-right:7px; }}
.ecard .m {{ font:14px var(--mono); color:var(--mut); margin-top:3px; }}
.ecard .m b {{ color:var(--sec); font-weight:600; }}
.chips {{ display:flex; gap:10px; justify-content:flex-end; flex:none; }}
.chip {{ font:600 14px var(--mono); background:var(--sup); border:1px solid var(--borde);
  border-radius:8px; padding:5px 12px; color:var(--mut); }}
.chip b {{ color:var(--txt); }}
.cards {{ overflow:hidden; display:flex; flex-direction:column; gap:8px; min-height:0; }}
.card {{ background:var(--sup); border:1px solid var(--borde); border-radius:10px;
  padding:6px 12px; cursor:pointer; flex:1 1 0; min-height:0; overflow:hidden; }}
.card:hover {{ border-color:var(--apoya); }}
.card.dim {{ opacity:.3; }}
.chead {{ font:600 15px var(--mono); color:var(--txt); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }}
.xchip {{ font:700 14px var(--mono); background:var(--sup2); border:1px solid var(--borde);
  border-radius:6px; padding:1px 8px; margin-left:6px; }}
.cmain {{ font:16px/1.25 var(--serif); color:var(--sec); margin-top:3px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }}
.csec {{ font:14px var(--mono); color:var(--mut); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }}
.track {{ height:7px; background:var(--sup2); border-radius:4px; overflow:hidden;
  display:flex; margin-top:5px; flex:none; }}
.fill {{ height:100%; }}
.p4top {{ display:flex; gap:16px; align-items:flex-start; flex:none; }}
.nums {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; flex:1; }}
.gbts {{ display:flex; gap:6px; flex:none; }}
.gbtn {{ background:var(--sup); color:var(--sec); border:1px solid var(--borde);
  border-radius:8px; padding:6px 12px; font:500 14px var(--mono); cursor:pointer; }}
.gbtn.on {{ color:var(--txt); border-color:var(--apoya); background:var(--sup2); }}
#p4body {{ flex:1; min-height:0; overflow:hidden; }}
.fl {{ font:600 14px var(--mono); color:var(--mut); text-transform:uppercase;
  letter-spacing:.05em; margin:8px 0 3px; }}
.coms {{ border-left:3px solid var(--amb-bo); padding-left:12px; }}
.coms .cq {{ font:italic 16px/1.3 var(--serif); color:var(--sec); margin:2px 0; }}
.mini {{ width:100%; height:130px; }} .mini2 {{ width:100%; height:150px; }}
#tt {{ position:fixed; display:none; max-width:420px; background:var(--sup2);
  border:1px solid var(--borde); border-radius:8px; padding:8px 12px; font-size:14px;
  color:var(--sec); pointer-events:none; z-index:9; }}
#tt li {{ margin:3px 0; font-style:italic; }}
@media (max-width:1100px) {{ .cols {{ grid-template-columns:1fr; }} .kpis {{ grid-template-columns:repeat(3,1fr); }} }}
</style>{plotly_js()}</head>
<body>
<header>
  <div class="brand"><span class="dot"></span><span class="bname">QualyData</span><span class="bset">/ {_esc(d['titulo_set'])}</span></div>
  <nav>{tabs}</nav>
  <button id="todo">[ver todo]</button>
</header>
<main>
<section class="paso" id="p1"><div class="kick">{KICKERS[0]}</div><h2 class="pt">{_esc(tit1)}</h2>{sub1_h}<div class="pc">
<div class="kpis">{kpis_h}</div>
<div class="barras" id="barras1"><div class="barr-head"><span class="h-lab">narrativa · entidades clave</span><span class="ph"><span><span class="sq" style="background:var(--apoya)"></span>apoya</span><span><span class="sq" style="background:var(--rechaza)"></span>rechaza</span></span><span class="nh">comentarios</span></div>{''.join(filas_p1)}<div class="barr mas" id="masrow" hidden></div></div>
<p class="pie">Los bandos se separan bien; decidir cuál apoya falla en algunas narrativas (macro-F1 0,65). · ruido: {100 * d['ruido']:.0f}% de los comentarios.</p>
</div></section>
<section class="paso" id="p2" hidden><div class="kick">{KICKERS[1]}</div><h2 class="pt" id="t2"></h2><div class="pc">
<div class="cols" style="grid-template-columns:1fr 320px"><div class="gwrap"><div id="g2" class="graf" style="height:100%"></div></div><div class="side" id="side2"></div></div>
<p class="pie">La fecha de origen es una estimación.</p></div></section>
<section class="paso" id="p3" hidden><div class="kick">{KICKERS[2]}</div><h2 class="pt">{_esc(_plural(d['n_grupos'], 'enjambre de cuentas actúa', 'enjambres de cuentas actúan'))} juntos mucho más que lo esperable por azar</h2><div class="pc">
<div class="chips" id="chips3"></div>
<div class="cols"><div class="gwrap"><div id="g3" class="graf" style="height:100%"></div></div><div class="cards" id="cards3"></div></div>
<p class="pie">Coordinación anómala detectada: una alerta para un analista, no prueba de manipulación. × = cohesión observada / cohesión con cuentas al azar.</p></div></section>
<section class="paso" id="p4" hidden><div class="kick" id="k4">{KICKERS[3]}</div><h2 class="pt" id="t4"></h2><div class="pc">
<div class="p4top"><div class="nums" id="nums4"></div><div class="gbts">{grupos_btn}</div></div>
<div id="p4body"></div>
<p class="pie">Coordinación anómala detectada: una alerta para un analista, no prueba de manipulación.</p></div></section>
<section class="paso" id="p5" hidden><div class="kick">{KICKERS[4]}</div><h2 class="pt">{_esc(titulo5)}</h2><div class="pc">
<div class="kpis" style="grid-template-columns:repeat(4,1fr)">{''.join(f'<div class="kpi"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in (x.split("|") for x in kpis5))}</div>
<div class="cols" style="grid-template-columns:1fr 300px"><div class="gwrap"><div id="g5" class="graf" style="height:100%"></div></div><div class="side" id="panel5"></div></div>
<p class="pie">{_esc(notas5)}</p></div></section>
</main>
<div id="tt"></div>
<script>
const D = {payload};
const $ = s => document.querySelector(s);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const CFG = {{ displaylogo:false, responsive:true, displayModeBar:false }};
const base = () => ({{ paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{ family:"'JetBrains Mono',monospace", size:14, color:css('--sec') }},
  margin:{{ l:56, r:16, t:36, b:46 }},
  xaxis:{{ gridcolor:css('--borde'), zeroline:false }},
  yaxis:{{ gridcolor:css('--borde'), zeroline:false, rangemode:'tozero' }} }});
const colorDe = i => i >= 0 ? css('--c' + (i % {len(PALETA)} + 1)) : css('--otras');
const fmtN = (x, dec = 0) => Number(x).toLocaleString('es-AR', {{ maximumFractionDigits:dec }});
const MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic'];
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
  L.showlegend = false;   // la leyenda vive en la barra lateral (tarjetas por emergente)
  L.shapes = shown.filter(l => D.origenes[l]).map(lab => ({{ type:'line',
    x0:D.origenes[lab].slice(0, 10), x1:D.origenes[lab].slice(0, 10), yref:'paper', y0:0, y1:1,
    line:{{ dash:'dash', width:2, color:colorDe(filaDe(lab).i) }} }}));
  L.annotations = shown.filter(l => D.origenes[l]).map((lab, k) => ({{
    x:D.origenes[lab].slice(0, 10), yref:'paper', y:1.0 - 0.09 * (k % 5), yanchor:'bottom',
    text:'origen ' + lab, showarrow:false, bgcolor:css('--bg'), borderpad:3,
    font:{{ size:14, family:"'JetBrains Mono',monospace", color:css('--sec') }} }}));
  Plotly.react('g2', tr, L, CFG);
  // barra lateral: una tarjeta por emergente (origen, pico con fecha, total)
  const fISO = iso => +iso.slice(8, 10) + '-' + MESES[+iso.slice(5, 7) - 1];
  const side = $('#side2');
  if (side) {{
    side.innerHTML = shown.map(lab => {{
      const f = filaDe(lab), s = D.serie_dia[lab] || [];
      let mx = 0, mi = 0;
      s.forEach((v, i) => {{ if (v > mx) {{ mx = v; mi = i; }} }});
      const tot = s.reduce((a, b) => a + b, 0);
      return '<div class="ecard' + (lab === foco ? ' on' : '') + '" data-nar="' + lab + '">' +
        '<div class="l"><span class="sw" style="background:' + colorDe(f.i) + '"></span>«' +
          (f.etiqueta.length > 55 ? f.etiqueta.slice(0, 55) + '…' : f.etiqueta) + '»</div>' +
        '<div class="m"><b>' + lab + '</b> · origen <b>' + f.origen_fecha + '</b></div>' +
        '<div class="m">pico <b>' + fmtN(mx) + '/día</b> el ' + fISO(D.x[mi]) +
          ' · total <b>' + fmtN(tot) + '</b></div></div>';
    }}).join('');
    side.querySelectorAll('.ecard').forEach(c => c.onclick = () => {{
      sel = {{ tipo:'nar', id:c.dataset.nar }}; pintarSel(); paso2();
    }});
  }}
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
  L.showlegend = true; L.legend = {{ orientation:'h', x:0.5, xanchor:'center',
    y:0.07, yanchor:'bottom', bgcolor:'rgba(11,13,18,.78)', borderpad:4,
    font:{{ size:14, family:"'JetBrains Mono',monospace" }} }};
  L.margin = {{ l:8, r:8, t:8, b:8 }};
  Plotly.react('g3', tr, L, CFG);
  const topG = D.tarjetas.reduce((a, b) => (b.nx_v > ((a || {{}}).nx_v || 0) ? b : a), null);
  const chips = $('#chips3');
  if (chips) chips.innerHTML =
    '<span class="chip">cuentas coordinadas: <b>' + fmtN(D.n_coord) + '</b></span>' +
    (topG ? '<span class="chip">× máximo: <b>' + fmtN(topG.nx_v, 1) + '×</b></span>' : '');
  const cont = $('#cards3');
  const et40 = e => e.etiqueta.length > 40 ? e.etiqueta.slice(0, 40) + '…' : e.etiqueta;
  cont.innerHTML = D.tarjetas.map(t =>
    '<div class="card' + (t === topG ? ' top' : '') + '" data-g="' + t.id + '"' +
    (t === topG ? ' style="border-color:' + colG[t.id] + '"' : '') + '>' +
    '<div class="chead">' + t.id + ' · ' + t.n + ' cuentas <span class="xchip">' + t.nx + '</span></div>' +
    (t.empuja.length ? '<div class="cmain">' + fmtN(t.empuja[0].pct, 1) + ' % en «' + et40(t.empuja[0]) + '»</div>' : '') +
    (t.empuja.length > 1 ? '<div class="csec">' +
      t.empuja.slice(1).map(e => e.id + ' ' + fmtN(e.pct, 1) + ' %').join(' · ') + '</div>' : '') +
    '<div class="track">' + t.empuja.map(e => '<div class="fill" style="width:' + Math.min(100, e.pct) +
      '%;background:' + colorDe(e.i) + '"></div>').join('') + '</div></div>').join('');
  cont.querySelectorAll('.card').forEach(c => c.onclick = () => {{ sel = {{ tipo:'grupo', id:c.dataset.g }}; irPaso(4, c.dataset.g); }});
}}

function paso4(gid) {{
  const t = D.tarjetas.find(x => x.id === gid) || D.tarjetas[0];
  if (!t) {{ $('#p4body').innerHTML = '<p class="pie">No se detectaron operaciones coordinadas.</p>'; return; }}
  document.querySelectorAll('.gbtn').forEach(b => b.classList.toggle('on', b.dataset.g === t.id));
  $('#k4').textContent = 'EVIDENCIA · ' + t.id;
  $('#t4').textContent = t.top_pct >= 50
    ? 'El grupo ' + t.id + ' publica el ' + fmtN(t.top_pct, 1) + ' % de sus comentarios en «' + t.top_et + '»'
    : 'Evidencia del grupo ' + t.id;
  $('#nums4').innerHTML =
    [[t.nx, 'co-actúan vs azar'],
     [fmtN(t.posts), 'publicaciones compartidas (≤ ' + fmtN(D.delta_min, 1) + ' min)'],
     [fmtN(t.respuestas), 'respuestas mutuas'],
     [fmtN(t.n_tags), 'hashtags raros' + (t.tags.length ? ': ' + t.tags.slice(0, 3).join(' ') : '')]]
    .map(x => '<div class="kpi"><div class="v">' + x[0] + '</div><div class="k">' + x[1] + '</div></div>').join('');
  $('#p4body').innerHTML =
    '<div class="cols" style="grid-template-columns:1fr 1fr;height:auto">' +
    '<div><div class="fl">% de comentarios por hora: ' + t.id + ' vs no coordinadas</div><div id="h4" class="mini"></div></div>' +
    '<div><div class="fl">«' + (t.post.titular || '').replace(/</g,'&lt;') + '» — un post, minuto a minuto</div><div id="g4" class="mini2"></div></div></div>' +
    '<div class="fl">comentarios del grupo</div><div class="coms">' +
    t.ej.slice(0, 3).map(x => '<div class="cq">«' + x.replace(/</g,'&lt;') + '»</div>').join('') + '</div>';
  const Lh = base(); Lh.margin = {{ l:70, r:60, t:4, b:52 }};
  Lh.xaxis = {{ tickmode:'array', tickvals:[0,3,6,9,12,15,18,21], tickfont:{{ size:14 }}, gridcolor:css('--borde'),
    title:{{ text:'hora local (−03:00)', font:{{ size:14 }}, standoff:28 }} }};
  Plotly.react('h4', [{{ z:[t.horas, D.horas_ng], y:[t.id, 'resto'], x:[...Array(24).keys()],
    type:'heatmap', showscale:true, colorscale:'Viridis',
    colorbar:{{ thickness:10, tickfont:{{ size:14 }}, title:{{ text:'% com.', font:{{ size:14 }} }} }} }}], Lh, CFG);
  const L4 = base(); L4.margin = {{ l:10, r:10, t:4, b:34 }};
  L4.xaxis = {{ title:{{ text:'minutos desde el primero', font:{{ size:14 }} }},
    gridcolor:css('--borde'), zeroline:false }};
  L4.yaxis = {{ type:'category', showticklabels:false, showgrid:false, ticks:'' }};
  Plotly.react('g4', [{{ x:t.post.pts.map(p => p.m), y:t.post.pts.map(p => p.c), mode:'markers',
    type:'scatter', marker:{{ size:7, color:css('--apoya') }},
    hovertemplate:'%{{y}} · +%{{x}} min<extra></extra>' }}], L4, CFG);
}}

function paso5() {{
  const et = Object.entries(D.etapas);
  const mx = et.reduce((a, b) => (b[1] > a[1] ? b : a), et[0] || ['', 0]);
  const tot = et.reduce((a, [, v]) => a + v, 0) || 1;
  const L = base(); L.margin.l = 130;
  L.xaxis.title = {{ text:'segundos' }}; L.yaxis.autorange = 'reversed';
  L.showlegend = false;
  if (et.length) L.annotations = [{{ x:mx[1], y:mx[0], text:'más lenta · ' + fmtN(100 * mx[1] / tot, 1) + ' % del total',
    showarrow:false, xanchor:'left', xshift:8, font:{{ size:14, color:css('--amb') }} }}];
  Plotly.react('g5', [{{ x:et.map(e => e[1]), y:et.map(e => e[0]), type:'bar',
    orientation:'h', marker:{{ color:et.map(([k]) => k === mx[0] ? css('--amb') : css('--apoya')) }},
    hovertemplate:'%{{y}}: %{{x:.1f}} s<extra></extra>' }}], L, CFG);
  const c = D.costo, P = [];
  if (c.comentarios_por_segundo != null)
    P.push([fmtN(c.comentarios_por_segundo), 'comentarios por segundo']);
  if (c.segundos_por_mil_comentarios != null)
    P.push([fmtN(c.segundos_por_mil_comentarios, 2) + ' s', 'segundos por mil comentarios']);
  if (c.proyeccion_1M_por_dia && c.proyeccion_1M_por_dia.horas != null)
    P.push([fmtN(Number(c.proyeccion_1M_por_dia.horas.toPrecision(2)), 2) + ' h', 'para 1M/día · estimación lineal']);
  const p5 = $('#panel5');
  if (p5) p5.innerHTML = P.map(x =>
    '<div class="kpi"><div class="v" style="font-size:34px">' + x[0] + '</div><div class="k">' + x[1] + '</div></div>').join('');
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
