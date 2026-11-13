"""e10 · Reporte HTML autocontenido para la demo: qué circula, desde cuándo y quién lo empuja.

Escribe reportes/<set>.html (no va en entrega/). Secciones:
  1. Encabezado con indicadores del set y de la corrida.
  2. Qué circula: tabla de narrativas (postura, origen, cuentas coordinadas, ejemplos).
  3. Desde cuándo: área apilada por día (top 8 + emergentes + "Otras") con el origen de las emergentes.
  4. Quién la empuja: una tarjeta por grupo coordinado (cohesión vs nulo, narrativas, ritmo
     horario y evidencia: posts compartidos, respuestas mutuas, hashtags raros).
  5. Costo y escala: costo.json aplanado y segundos por etapa.
  6. Cómo leer esto.
Usa plotly.js embebido si el paquete plotly está instalado; si no, lo toma del CDN.
"""
from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qd import costo as qcosto
from qd.config import obtener

log = logging.getLogger("qd")
SALIDAS: list = []
TZ = timezone(timedelta(hours=-3))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
          "#0f9fb5", "#8c5a3c", "#8a9a1b", "#b04fc0"]
SERIES_OSC = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767",
              "#22b3c9", "#a87252", "#a3b52a", "#c66ad6"]
GRIS = "#9a9890"
HASHTAG_RE = re.compile(r"#\w{3,}")


def _f(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=TZ).strftime("%Y-%m-%d %H:%M")


def _fecha(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=TZ).strftime("%Y-%m-%d")


def _esc(x) -> str:
    return html.escape(str(x))


def _tabla(df: pd.DataFrame) -> str:
    cab = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    filas = "".join("<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in fila) + "</tr>" for fila in df.itertuples(index=False))
    return f'<div class="tabla"><table><thead><tr>{cab}</tr></thead><tbody>{filas}</tbody></table></div>'


# --- evidencia de grupo (funciones puras, testeables sin almacén)

def posts_compartidos(g: pd.DataFrame, delta_s: int) -> int:
    """# publicaciones donde >=2 miembros del grupo comentaron a <= delta_s una de otra.
    g: comentarios del grupo con columnas cuenta_id, publicacion_id, ts."""
    n = 0
    for _, sub in g.groupby("publicacion_id", sort=False):
        ctas = sub["cuenta_id"].to_numpy()
        ts = sub["ts"].to_numpy()
        orden = np.argsort(ts, kind="stable")
        for i in range(1, len(orden)):
            if ts[orden[i]] - ts[orden[i - 1]] <= delta_s and ctas[orden[i]] != ctas[orden[i - 1]]:
                n += 1
                break
    return n


def respuestas_mutuas(g: pd.DataFrame) -> int:
    """# comentarios del grupo que responden a un comentario de OTRO miembro.
    g: comentarios del grupo con columnas comentario_id, cuenta_id, respuesta_a."""
    dueño = g.set_index("comentario_id")["cuenta_id"]
    r = g[g["respuesta_a"] != ""]
    destino = r["respuesta_a"].map(dueño)
    return int((destino.notna() & (destino != r["cuenta_id"])).sum())


def mapa_hashtags(df: pd.DataFrame) -> dict:
    """{hashtag en minúscula: cuentas que lo usaron} sobre columnas cuenta_id, texto."""
    uso = {}
    for cta, txt in zip(df["cuenta_id"], df["texto"]):
        for tag in set(HASHTAG_RE.findall(str(txt).lower())):
            uso.setdefault(tag, set()).add(cta)
    return uso


def hashtags_raros(uso_g: dict, uso_total: dict, umbral: int, min_miembros: int = 2) -> list:
    """Hashtags usados por >= min_miembros del grupo y por <= umbral cuentas en todo el set."""
    return sorted(t for t, c in uso_g.items()
                  if len(c) >= min_miembros and len(uso_total.get(t, ())) <= umbral)


def _aplanar(d: dict, pref: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _aplanar(v, f"{pref}{k}.")
        else:
            yield f"{pref}{k}", v


def datos_reporte(ctx) -> dict:
    db, cfg = ctx.db, ctx.cfg
    com = db.leer("com")[["comentario_id", "cuenta_id", "publicacion_id", "ts", "texto", "respuesta_a"]]
    asig = db.leer("asig")[["comentario_id", "narrativa", "postura", "conf"]]
    narr = (db.leer("narr") if db.existe("narr") else
            pd.DataFrame(columns=["narrativa", "titulo", "frases_clave", "ts_origen", "n"]))
    grupos = db.leer("grupos") if db.existe("grupos") else pd.DataFrame(columns=["cuenta_id", "grupo"])
    grupos = grupos[grupos["cuenta_id"] != ""]
    gdiag = db.leer("grupos_diag") if db.existe("grupos_diag") else pd.DataFrame()
    gdiag = gdiag[gdiag["grupo"] != ""] if len(gdiag) else gdiag

    df = com.merge(asig, on="comentario_id").merge(grupos, on="cuenta_id", how="left")
    df["grupo"] = df["grupo"].fillna("")
    nar = df[df["narrativa"] != ""]
    t0, t1 = int(com["ts"].min()), int(com["ts"].max())
    span = max(t1 - t0, 1)

    # --- narrativas mostradas (filtro reporte.min_frac) y sus colores
    conteo = nar["narrativa"].value_counts()
    min_frac = float(obtener(cfg, "reporte.min_frac", 0.001))
    orden = conteo[conteo >= min_frac * len(com)].index.tolist()
    ocultas = conteo[~conteo.index.isin(orden)]
    top = orden[:8]
    titulos = dict(zip(narr["narrativa"], narr["titulo"])) if len(narr) else {}
    frases = (dict(zip(narr["narrativa"], narr["frases_clave"]))
              if len(narr) and "frases_clave" in narr else {})

    # ts_origen y flag "emergente" por narrativa (una pasada; se reusa en la tabla)
    idx_o = narr.set_index("narrativa")["ts_origen"] if len(narr) else pd.Series(dtype=float)
    ts_o_map, emergente = {}, {}
    for lab in orden:
        g = nar[nar["narrativa"] == lab]
        t_o = (int(idx_o[lab]) if lab in idx_o.index and pd.notna(idx_o[lab])
               else int(g["ts"].min()))
        ts_o_map[lab] = t_o
        emergente[lab] = (t_o - t0) > 0.1 * span
    emerg = {lab: ts_o_map[lab] for lab in orden if emergente[lab]}
    mostradas = top + [lab for lab in orden if emergente[lab] and lab not in top]
    color_n = {lab: mostradas.index(lab) % len(SERIES) if lab in mostradas else -1
               for lab in orden}

    # --- serie por día
    nar_v = nar[nar["narrativa"].isin(orden)].copy()
    nar_v["serie"] = np.where(nar_v["narrativa"].isin(mostradas), nar_v["narrativa"], "Otras")
    nar_v["b"] = (nar_v["ts"] - t0) // 86400
    nb = int((t1 - t0) // 86400) + 1
    tab = (nar_v.groupby(["b", "serie"]).size().unstack(fill_value=0)
           .reindex(range(nb), fill_value=0))
    x = [_fecha(t0 + i * 86400) for i in range(nb)]
    series = [{"id": s, "titulo": titulos.get(s, "Otras narrativas" if s == "Otras" else s),
               "y": tab[s].astype(int).tolist() if s in tab else [0] * nb,
               "ci": color_n.get(s, -1)}
              for s in mostradas + (["Otras"] if "Otras" in tab.columns else [])]

    # --- tabla de narrativas
    filas, origenes = [], []
    for lab in orden:
        g = nar[nar["narrativa"] == lab]
        pct = g["postura"].value_counts(normalize=True)
        ej = (g.sort_values("conf", ascending=False).drop_duplicates("texto")["texto"]
              .head(3).map(lambda s: str(s)[:200]).tolist())
        filas.append({"id": lab, "titulo": titulos.get(lab, lab), "i": color_n[lab], "n": len(g),
                      "frases": str(frases.get(lab, "")).replace("|", " · "),
                      "apoya": float(pct.get("apoya", 0.0)), "rechaza": float(pct.get("rechaza", 0.0)),
                      "origen": _f(ts_o_map[lab]), "emergente": emergente[lab],
                      "coord": float((g["grupo"] != "").mean()), "ejemplos": ej})
        if emergente[lab]:
            origenes.append({"x": _f(ts_o_map[lab]), "id": lab, "i": color_n[lab],
                             "titulo": titulos.get(lab, lab)[:30]})

    # --- tarjetas de grupos coordinados
    delta = int(obtener(cfg, "coord.delta_s", 450))
    umbral_tag = max(int(obtener(cfg, "coord.min_size", 5)),
                     float(obtener(cfg, "coord.hashtag_max_frac", 0.01)) * df["cuenta_id"].nunique())
    uso_total = mapa_hashtags(df[["cuenta_id", "texto"]]) if len(grupos) else {}
    diag = gdiag.set_index("grupo") if len(gdiag) else pd.DataFrame()
    tarjetas = []
    for gid in grupos["grupo"].value_counts().index.tolist():
        ctas = set(grupos.loc[grupos["grupo"] == gid, "cuenta_id"])
        g = df[df["cuenta_id"].isin(ctas)]
        gn = g[g["narrativa"] != ""]
        empuja = []
        for lab, cnt in gn["narrativa"].value_counts().head(3).items():
            sub = gn[gn["narrativa"] == lab]
            posturas = sub["postura"].mode()
            empuja.append({"id": lab, "titulo": titulos.get(lab, lab), "i": color_n.get(lab, -1),
                           "n": int(cnt), "pct": 100 * cnt / max(len(g), 1),
                           "postura": posturas.iloc[0] if len(posturas) else "-"})
        nx_txt = "-"
        if len(diag) and gid in diag.index:
            co, cn = diag.loc[gid, "coh_obs"], diag.loc[gid, "coh_nulo"]
            if pd.notna(co) and pd.notna(cn):
                nx_txt = f"{co / cn:.1f}×".replace(".", ",") if cn else "mucho más"
        horas = [0] * 24
        for h, c in (((g["ts"] - 3 * 3600) // 3600) % 24).value_counts().items():
            horas[int(h)] = int(c)
        tags = hashtags_raros(mapa_hashtags(g[["cuenta_id", "texto"]]), uso_total, umbral_tag)
        temprano = [{"id": lab,
                     "k": int(((gn["narrativa"] == lab) & (gn["ts"] >= ts_o)
                               & (gn["ts"] <= ts_o + 6 * 3600)).sum())}
                    for lab, ts_o in emerg.items()]
        temprano = [t for t in temprano if t["k"]]
        tarjetas.append({"id": gid, "n": len(ctas), "nx": nx_txt, "empuja": empuja, "horas": horas,
                         "posts": posts_compartidos(g[["cuenta_id", "publicacion_id", "ts"]], delta),
                         "respuestas": respuestas_mutuas(g[["comentario_id", "cuenta_id", "respuesta_a"]]),
                         "tags": tags, "temprano": temprano})

    # --- costo (almacenamiento medido acá: pipeline.main lo mide después de e10)
    costo = dict(ctx.costo.a_dict())
    costo["almacenamiento_mb"] = qcosto.medir_almacenamiento(ctx.datos, ctx.work)
    etapas = costo.get("desglose_segundos", {})
    usd_mil = costo.get("usd_computo_por_mil")
    if usd_mil is not None:
        usd_mil += costo.get("usd_por_mil_comentarios") or 0

    return {"x": x, "series": series, "origenes": origenes, "narr": filas,
            "ocultas": {"n": int(len(ocultas)), "comentarios": int(ocultas.sum())},
            "tarjetas": tarjetas,
            "costo": [{"k": k, "v": v} for k, v in _aplanar(costo)],
            "etapas": {"etapa": list(etapas), "seg": [float(v) for v in etapas.values()]},
            "total": len(com), "dias": max(1, round(span / 86400)),
            "emergentes": len(emerg), "ruido": float((df["narrativa"] == "").mean()),
            "n_coord": int(len(grupos)),
            "n_grupos": int(grupos["grupo"].nunique()) if len(grupos) else 0,
            "segundos": float(costo.get("segundos", 0)), "usd_mil": usd_mil,
            "periodo": f"{_f(t0)} → {_f(t1)}"}


def plotly_js() -> str:
    try:
        from plotly.offline import get_plotlyjs

        return f"<script>{get_plotlyjs()}</script>"
    except Exception:  # noqa: BLE001
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def _num(x, dec=None) -> str:
    """Miles con punto y decimales con coma, solo sobre el número."""
    return (f"{x:,.{dec}f}" if dec is not None else f"{x:,}").replace(",", " ").replace(".", ",").replace(" ", ".")


def _plural(n: int, sing: str, plur: str) -> str:
    return f"{n} {sing if n == 1 else plur}"


def _var_color(i: int) -> str:
    return f"var(--series-{(i % len(SERIES)) + 1})" if i >= 0 else "var(--otras)"


def _tabla_narr(filas: list) -> str:
    trs = []
    for r in filas:
        ej = "".join(f"<li>{_esc(t)}</li>" for t in r["ejemplos"])
        badge = ' <span class="badge">emergente</span>' if r["emergente"] else ""
        trs.append(
            f'<tr><td><span class="sw" style="background:{_var_color(r["i"])}"></span></td>'
            f'<td><b>{_esc(r["id"])}</b> · {_esc(r["titulo"])}'
            f'<div class="fr">{_esc(r["frases"])}</div>'
            f'<details><summary>ejemplos</summary><ul class="ej">{ej}</ul></details></td>'
            f'<td>{_num(r["n"])}</td>'
            f'<td><div class="post"><span class="ap" style="width:{100 * r["apoya"]:.0f}%"></span>'
            f'<span class="re" style="width:{100 * r["rechaza"]:.0f}%"></span></div>'
            f'<div class="fr">{100 * r["apoya"]:.0f}% apoya · {100 * r["rechaza"]:.0f}% rechaza</div></td>'
            f'<td>{_esc(r["origen"])}{badge}</td>'
            f'<td>{_num(100 * r["coord"], 1)}%</td></tr>')
    return ('<div class="tabla"><table><thead><tr><th></th><th>Narrativa</th><th>Comentarios</th>'
            '<th>Postura</th><th>Origen</th><th>% coord.</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def _tarjetas(tarjetas: list, delta_s: int) -> str:
    out = []
    for t in tarjetas:
        nx = (f' · co-actúan <b>{_esc(t["nx"])}</b> más que esas mismas cuentas al azar'
              if t["nx"] != "-" else "")
        empuja = "".join(
            f'<li><span class="sw" style="background:{_var_color(e["i"])}"></span>'
            f'{_esc(e["id"])} · {_esc(e["titulo"])} — {_num(e["pct"], 1)}% · {_esc(e["postura"])}</li>'
            for e in t["empuja"])
        temprano = "".join(f'<span class="badge">temprano en {_esc(x["id"])} (emergente): '
                           f'{x["k"]} comentarios en las primeras 6 h</span> ' for x in t["temprano"])
        tags = _plural(len(t["tags"]), "hashtag raro", "hashtags raros") + (
            f': {" ".join(_esc(x) for x in t["tags"][:3])}' if t["tags"] else "")
        out.append(f'''<div class="card">
  <div class="card-h"><b>{_esc(t["id"])}</b> · {t["n"]} cuentas{nx}</div>
  <div class="fr">empuja (top 3 por % de sus comentarios)</div><ul class="emp">{empuja or "<li>-</li>"}</ul>
  <div id="h_{_esc(t["id"])}" class="heat"></div>
  <div class="fr">evidencia</div>
  <ul class="ev"><li>{_plural(t["posts"], "publicación compartida", "publicaciones compartidas")} (≥2 miembros a ≤ {delta_s} s)</li>
  <li>{_plural(t["respuestas"], "respuesta mutua", "respuestas mutuas")}</li><li>{tags}</li></ul>
  {temprano}
</div>''')
    return '<div class="cards">' + "".join(out) + '</div>'


def html_reporte(d: dict, titulo: str) -> str:
    usd = f'USD {_num(d["usd_mil"], 5)}/1.000 com.' if d["usd_mil"] is not None else "sin USD estimado"
    kpis = [("Comentarios", _num(d["total"])),
            ("Días", str(d["dias"])),
            ("Narrativas", str(len(d["narr"]))),
            ("Emergentes", str(d["emergentes"])),
            (f"grupos coord. · {_num(d['n_coord'])} cuentas", str(d["n_grupos"])),
            (usd, f"{d['segundos']:.0f} s")]
    tiles = "".join(f'<div class="kpi"><div class="v">{_esc(v)}</div><div class="k">{_esc(k)}</div></div>' for k, v in kpis)
    heats = {t["id"]: t["horas"] for t in d["tarjetas"]}
    payload = json.dumps({"x": d["x"], "series": d["series"], "origenes": d["origenes"],
                          "etapas": d["etapas"], "heat": heats}, ensure_ascii=False)
    ocultas = (f'<p class="nota">{d["ocultas"]["n"]} narrativas chicas '
               f'({_num(d["ocultas"]["comentarios"])} comentarios) no se muestran</p>'
               if d["ocultas"]["n"] else "")
    costo_tab = _tabla(pd.DataFrame({"Clave": [r["k"] for r in d["costo"]],
                                     "Valor": [r["v"] for r in d["costo"]]}))
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(titulo)}</title>
<style>
.viz-root {{ color-scheme: light; --surface-1:#fcfcfb; --surface-2:#f2f1ed; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --text-muted:#7a7972; --grid:#e4e3de; --borde:#dddcd6; --apoya:#1baf7a; --rechaza:#e34948;
  {''.join(f'--series-{i + 1}:{h};' for i, h in enumerate(SERIES))} --otras:{GRIS}; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232321; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#9a998f;
  --grid:#30302d; --borde:#3a3a37; --apoya:#199e70; --rechaza:#e66767;
  {''.join(f'--series-{i + 1}:{h};' for i, h in enumerate(SERIES_OSC))} }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
.viz-root {{ background:var(--surface-1); color:var(--text-primary); min-height:100vh; padding:28px 32px 60px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 6px; }}
.sub {{ color:var(--text-secondary); margin:0 0 18px; }} .nota {{ color:var(--text-muted); font-size:13px; margin:4px 0 10px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
.kpi {{ background:var(--surface-2); border-radius:10px; padding:12px 14px; }}
.kpi .v {{ font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }} .kpi .k {{ color:var(--text-secondary); font-size:12px; }}
.graf {{ width:100%; height:400px; }} .graf.chico {{ height:300px; }}
.tabla {{ overflow-x:auto; border:1px solid var(--borde); border-radius:10px; margin-top:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--borde); vertical-align:top; }}
th {{ color:var(--text-secondary); font-weight:600; background:var(--surface-2); position:sticky; top:0; }}
td {{ font-variant-numeric:tabular-nums; }}
.sw {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:2px; }}
.fr {{ color:var(--text-muted); font-size:12px; margin-top:2px; }}
.post {{ display:flex; height:9px; width:120px; border-radius:5px; overflow:hidden; background:var(--surface-2); }}
.post .ap {{ background:var(--apoya); }} .post .re {{ background:var(--rechaza); }}
.badge {{ display:inline-block; background:var(--surface-2); border:1px solid var(--borde); border-radius:8px;
  padding:1px 7px; font-size:11px; color:var(--text-secondary); margin-top:3px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:12px; margin-top:8px; }}
.card {{ background:var(--surface-2); border:1px solid var(--borde); border-radius:10px; padding:12px 14px; }}
.card-h {{ font-size:14px; margin-bottom:6px; }}
ul {{ margin:4px 0 8px; padding-left:18px; }} li {{ margin:2px 0; }}
.heat {{ width:100%; height:90px; }}
details summary {{ cursor:pointer; color:var(--text-muted); font-size:12px; }}
ul.ej {{ font-style:italic; color:var(--text-secondary); }}
</style>{plotly_js()}</head>
<body><div class="viz-root">
<h1>{_esc(titulo)}</h1>
<p class="sub">Período {_esc(d['periodo'])} · qué circula, desde cuándo y quién lo empuja</p>
<div class="kpis">{tiles}</div>

<h2>1 · Qué circula</h2>
<p class="nota">Narrativas por tamaño. La barra reparte postura (apoya en verde, rechaza en rojo); el resto no tiene bando claro.</p>
{_tabla_narr(d['narr'])}
{ocultas}

<h2>2 · Desde cuándo</h2>
<p class="nota">Comentarios por día y narrativa (las 8 más grandes y todas las emergentes; el resto en «Otras»). Las líneas punteadas marcan el origen estimado de cada emergente, en el color de su serie.</p>
<div id="g_tiempo" class="graf"></div>

<h2>3 · Quién la empuja</h2>
<p class="nota">Una tarjeta por grupo coordinado: cuántas cuentas, cuánto más co-actúan que esas mismas cuentas al azar, qué narrativas empujan y su evidencia. Misma escala de color en todas las tarjetas.</p>
{_tarjetas(d['tarjetas'], d['delta_s']) if d['tarjetas'] else '<p class="nota">No se detectaron operaciones coordinadas.</p>'}

<h2>4 · Costo y escala</h2>
<div id="g_costo" class="graf chico"></div>
{costo_tab}

<h2>5 · Cómo leer esto</h2>
<ul class="nota">
<li>Postura macro-F1 ≈ 0,65 en práctica: los bandos de cada narrativa salen bien separados; lo que a veces falla es cuál bando «rechaza», y eso invierte la narrativa entera.</li>
<li>Coordinación = cuentas que co-actúan repetidamente por encima de lo esperable por azar. NO es prueba de manipulación ni de intención: puede ser una comunidad organizada legítima.</li>
<li>El origen de cada narrativa es una estimación (el punto donde aparece por primera vez como conjunto).</li>
<li>Ruido = comentarios genéricos sin narrativa ({100 * d['ruido']:.0f}% de este set).</li>
</ul>
</div>
<script>
const D = {payload};
const css = n => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(n).trim();
const base = () => ({{ paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{ family:'system-ui, sans-serif', size:12, color:css('--text-secondary') }},
  margin:{{ l:56, r:16, t:10, b:46 }}, hoverlabel:{{ bgcolor:css('--surface-2'), font:{{ color:css('--text-primary') }} }},
  xaxis:{{ gridcolor:css('--grid'), zeroline:false, linecolor:css('--grid') }},
  yaxis:{{ gridcolor:css('--grid'), zeroline:false, rangemode:'tozero' }} }});
const color = (ci, id) => id === 'Otras' ? css('--otras') : css('--series-' + (ci + 1));
function dibujar() {{
  const sup = css('--surface-1');
  const tr = D.series.map(s => ({{ x:D.x, y:s.y, name:s.id + ' · ' + s.titulo.slice(0, 42), type:'scatter', mode:'lines',
    stackgroup:'uno', line:{{ width:1.5, color:sup }}, fillcolor:color(s.ci, s.id),
    hovertemplate:'<b>' + s.id + '</b> ' + s.titulo.replace(/</g, '&lt;') + '<br>%{{x}}: %{{y}} comentarios<extra></extra>' }}));
  const L1 = base();
  L1.yaxis.title = {{ text:'comentarios por día' }}; L1.hovermode = 'x unified'; L1.showlegend = true;
  L1.legend = {{ orientation:'h', y:-0.18, font:{{ color:css('--text-primary') }} }}; L1.margin.b = 110;
  L1.margin.t = Math.max(28, 14 * Math.ceil(D.origenes.length / 7) + 14);  /* lugar para las anotaciones de origen */
  L1.shapes = D.origenes.map(o => ({{ type:'line', x0:o.x, x1:o.x, yref:'paper', y0:0, y1:1,
    line:{{ dash:'dash', width:2.5, color:css('--series-' + (o.i + 1)) }} }}));
  L1.annotations = D.origenes.map((o, k) => ({{ x:o.x, yref:'paper', y:1.0 - 0.08 * (k % 7),
    text:'origen ' + o.id + ' · ' + o.titulo, showarrow:false, yanchor:'bottom',
    bgcolor:css('--surface-1'), borderwidth:0, borderpad:2,
    font:{{ size:11, color:css('--text-secondary') }} }}));
  const cfg = {{ displaylogo:false, responsive:true, displayModeBar:false }};
  Plotly.react('g_tiempo', tr, L1, cfg);

  const zmax = Math.max(1, ...Object.values(D.heat).flat());
  for (const gid in D.heat) {{
    const Lh = base(); Lh.margin = {{ l:10, r:70, t:4, b:30 }};
    Lh.xaxis = {{ title:{{ text:'hora local (−03:00)', font:{{ size:10 }} }}, tickfont:{{ size:9 }},
      tickmode:'array', tickvals:[0, 3, 6, 9, 12, 15, 18, 21], gridcolor:css('--grid'), zeroline:false }};
    Lh.yaxis = {{ visible:false }};
    Plotly.react('h_' + gid, [{{ z:[D.heat[gid]], x:[...Array(24).keys()], y:[''],
      type:'heatmap', showscale:true, zmin:0, zmax:zmax,
      colorbar:{{ thickness:8, tickfont:{{ size:9 }}, title:{{ text:'comentarios', font:{{ size:9 }} }} }},
      colorscale:'Viridis', hovertemplate:'%{{x}}h: %{{z}} comentarios<extra></extra>' }}], Lh, cfg);
  }}

  const L3 = base();
  L3.margin.l = 120; L3.xaxis.title = {{ text:'segundos' }}; L3.yaxis.autorange = 'reversed'; L3.showlegend = true;
  L3.legend = {{ orientation:'h', y:-0.12, font:{{ color:css('--text-primary') }} }};
  Plotly.react('g_costo', [{{ x:D.etapas.seg, y:D.etapas.etapa, type:'bar', orientation:'h',
    name:'segundos por etapa', marker:{{ color:css('--series-1') }},
    hovertemplate:'%{{y}}: %{{x:.1f}} s<extra></extra>' }}], L3, cfg);
}}
dibujar();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', dibujar);
</script></body></html>"""


def correr(ctx) -> None:
    if not obtener(ctx.cfg, "reporte.activo", True):
        return
    d = datos_reporte(ctx)
    d["delta_s"] = int(obtener(ctx.cfg, "coord.delta_s", 450))
    destino = Path(obtener(ctx.cfg, "reporte.dir", "reportes")) / f"{ctx.datos.name}.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html_reporte(d, f"QualyData · narrativas y coordinación · {ctx.datos.name}"), encoding="utf-8")
    log.info("e10: reporte -> %s", destino)
