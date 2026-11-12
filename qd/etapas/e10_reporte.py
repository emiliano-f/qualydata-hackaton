"""e10 · Reporte HTML autocontenido para la demo: qué circula, desde cuándo y quién lo empuja.

Escribe reportes/<set>.html (no va en entrega/). Tres vistas:
  1. Narrativas en el tiempo (área apilada cada 6 h, top 8 + "Otras") con el origen de las emergentes,
     y la tabla de narrativas: título, volumen, % rechazo, origen y qué grupo coordinado la empuja.
  2. Operaciones coordinadas: red de cuentas (aristas = co-acción repetida) y tabla por grupo.
  3. Costo y escala: indicadores de costo.json y segundos por etapa.
Usa plotly.js embebido si el paquete plotly está instalado; si no, lo toma del CDN.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qd.config import obtener

log = logging.getLogger("qd")
SALIDAS: list = []
TZ = timezone(timedelta(hours=-3))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_OSC = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
GRIS = "#9a9890"


def _f(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=TZ).strftime("%Y-%m-%d %H:%M")


def _esc(x) -> str:
    return html.escape(str(x))


def _tabla(df: pd.DataFrame) -> str:
    cab = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    filas = "".join("<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in fila) + "</tr>" for fila in df.itertuples(index=False))
    return f'<div class="tabla"><table><thead><tr>{cab}</tr></thead><tbody>{filas}</tbody></table></div>'


def datos_reporte(db, costo: dict, cfg: dict) -> dict:
    com = db.leer("com")[["comentario_id", "cuenta_id", "ts", "respuesta_a"]]
    asig = db.leer("asig")[["comentario_id", "narrativa", "postura"]]
    narr = db.leer("narr") if db.existe("narr") else pd.DataFrame(columns=["narrativa", "titulo", "ts_origen", "n"])
    grupos = db.leer("grupos") if db.existe("grupos") else pd.DataFrame(columns=["cuenta_id", "grupo"])
    grupos = grupos[grupos["cuenta_id"] != ""]
    df = com.merge(asig, on="comentario_id")
    df = df.merge(grupos, on="cuenta_id", how="left")
    df["grupo"] = df["grupo"].fillna("")
    nar = df[df["narrativa"] != ""]
    t0, t1 = int(com["ts"].min()), int(com["ts"].max())

    # --- serie temporal (sin las narrativas diminutas de reporte.min_frac)
    conteo = nar["narrativa"].value_counts()
    min_frac = float(obtener(cfg, "reporte.min_frac", 0.001))
    orden = conteo[conteo >= min_frac * len(com)].index.tolist()
    ocultas = conteo[~conteo.index.isin(orden)]
    top = orden[:8]
    nar_v = nar[nar["narrativa"].isin(orden)]
    nar_v = nar_v.assign(serie=np.where(nar_v["narrativa"].isin(top), nar_v["narrativa"], "Otras"))
    paso = 6 * 3600
    nar_v = nar_v.assign(b=(nar_v["ts"] - t0) // paso)
    tab = nar_v.groupby(["b", "serie"]).size().unstack(fill_value=0).reindex(range(int((t1 - t0) // paso) + 1),
                                                                            fill_value=0)
    titulos = dict(zip(narr["narrativa"], narr["titulo"])) if len(narr) else {}
    x = [_f(t0 + i * paso) for i in tab.index]
    series = [{"id": s, "titulo": titulos.get(s, "Otras narrativas" if s == "Otras" else s),
               "y": tab[s].astype(int).tolist() if s in tab else [0] * len(x)}
              for s in top + (["Otras"] if "Otras" in tab else [])]

    # --- tabla de narrativas
    span = t1 - t0
    # Quién la empuja: grupo con mayor lift = share en la narrativa / share en todas las narrativas
    base_g = nar[nar["grupo"] != ""]["grupo"].value_counts() / max(len(nar), 1)
    filas, origenes = [], []
    for lab in orden:
        g = nar[nar["narrativa"] == lab]
        fila_n = narr[narr["narrativa"] == lab]
        ts_o = int(fila_n["ts_origen"].iloc[0]) if len(fila_n) and pd.notna(fila_n["ts_origen"].iloc[0]) else int(g["ts"].min())
        emergente = (ts_o - t0) > 0.1 * span
        gc = g[g["grupo"] != ""]["grupo"].value_counts()
        empuja = "-"
        if len(gc):
            lift = (gc / len(g)) / base_g.reindex(gc.index).fillna(1e-9)
            ok = gc[(lift >= 3) & (gc >= 20)]
            if len(ok):
                gid = lift[ok.index].idxmax()
                empuja = (f"{gid} · {100 * gc[gid] / len(g):.1f}% de los comentarios, "
                          f"{lift[gid]:.0f}× lo esperable").replace(".", ",")
        filas.append({"Narrativa": lab, "Título": titulos.get(lab, lab), "Comentarios": len(g),
                      "% rechaza": f"{100 * (g['postura'] == 'rechaza').mean():.0f}%",
                      "Origen": _f(ts_o) + (" · emergente" if emergente else ""),
                      "% de cuentas coordinadas": f"{100 * (g['grupo'] != '').mean():.1f}%",
                      "Quién la empuja": empuja})
        if emergente:
            origenes.append({"x": _f(ts_o), "id": lab, "i": top.index(lab) if lab in top else -1,
                             "titulo": titulos.get(lab, lab)[:30]})

    # --- red y grupos
    nodos, aristas, filas_g = [], [], []
    if len(grupos):
        import networkx as nx

        pares = db.leer("coord_pares") if db.existe("coord_pares") else pd.DataFrame(columns=["u", "v"])
        g2 = dict(zip(grupos["cuenta_id"], grupos["grupo"]))
        pares = pares[pares["u"].isin(g2) & pares["v"].isin(g2)]
        pares = pares[pares["u"].map(g2) == pares["v"].map(g2)]
        G = nx.Graph()
        G.add_nodes_from(g2)
        G.add_edges_from(zip(pares["u"], pares["v"]))
        orden_g = grupos["grupo"].value_counts().index.tolist()
        pos = {}
        cols = int(np.ceil(np.sqrt(len(orden_g))))
        for i, gid in enumerate(orden_g):
            sub = G.subgraph([c for c, x in g2.items() if x == gid])
            p = nx.spring_layout(sub, seed=42) if len(sub) > 1 else {n: np.zeros(2) for n in sub}
            esc = 0.35 + 0.08 * np.sqrt(len(sub))
            for n, xy in p.items():
                pos[n] = (float(xy[0]) * esc + 2.2 * (i % cols), float(xy[1]) * esc - 2.2 * (i // cols))
        act = df.groupby("cuenta_id").size()
        cta = db.leer("cta").set_index("cuenta_id") if db.existe("cta") else pd.DataFrame()
        for n in G.nodes:
            nodos.append({"id": n, "g": g2[n], "x": pos[n][0], "y": pos[n][1], "n": int(act.get(n, 0))})
        for u, v in G.edges:
            aristas.append([pos[u][0], pos[u][1], pos[v][0], pos[v][1]])
        for gid in orden_g:
            ctas = [c for c, x in g2.items() if x == gid]
            g = df[df["cuenta_id"].isin(ctas)]
            gn = g[g["narrativa"] != ""]
            principal = gn.groupby(["narrativa", "postura"]).size().sort_values(ascending=False)
            if len(principal):
                (lab, post), cnt = principal.index[0], principal.iloc[0]
                txt = f"{titulos.get(lab, lab)} ({post}, {100 * cnt / len(gn):.0f}%)"
            else:
                txt = "-"
            edad = pd.to_numeric(cta.reindex(ctas)["edad_dias"], errors="coerce").median() if len(cta) else np.nan
            horas = ((g["ts"] - 3 * 3600) // 3600 % 24).value_counts()
            filas_g.append({"Grupo": gid, "Cuentas": len(ctas), "Comentarios": len(g), "Empuja": txt,
                            "Antigüedad mediana": f"{edad:.0f} días" if edad == edad else "-",
                            "Horas pico": ", ".join(f"{int(h)} h" for h in horas.index[:3]),
                            "Activo": f"{_f(g['ts'].min())} → {_f(g['ts'].max())}" if len(g) else "-"})

    etapas = costo.get("desglose_segundos", {})
    return {"x": x, "series": series, "origenes": origenes, "narr": pd.DataFrame(filas),
            "ocultas": {"n": int(len(ocultas)), "comentarios": int(ocultas.sum())},
            "nodos": nodos, "aristas": aristas, "grupos": pd.DataFrame(filas_g), "costo": costo,
            "etapas": {"etapa": list(etapas), "seg": [float(v) for v in etapas.values()]},
            "total": len(com), "ruido": float((df["narrativa"] == "").mean()),
            "n_coord": int(len(grupos)), "n_grupos": int(grupos["grupo"].nunique()) if len(grupos) else 0,
            "periodo": f"{_f(t0)} → {_f(t1)}"}


def plotly_js() -> str:
    try:
        from plotly.offline import get_plotlyjs

        return f"<script>{get_plotlyjs()}</script>"
    except Exception:  # noqa: BLE001
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def html_reporte(d: dict, titulo: str) -> str:
    c = d["costo"]
    kpis = [("Comentarios", f"{d['total']:,}".replace(",", ".")), ("Narrativas", str(len(d["narr"]))),
            ("Ruido", f"{100 * d['ruido']:.0f}%"), ("Cuentas coordinadas", f"{d['n_coord']} en {d['n_grupos']} grupos"),
            ("Proceso", f"{c.get('segundos', 0):.0f} s"), ("USD por 1.000 comentarios", f"{c.get('usd_por_mil_comentarios', 0):.5f}")]
    tiles = "".join(f'<div class="kpi"><div class="v">{_esc(v)}</div><div class="k">{_esc(k)}</div></div>' for k, v in kpis)
    payload = json.dumps({k: d[k] for k in ("x", "series", "origenes", "nodos", "aristas", "etapas")}, ensure_ascii=False)
    costo_tab = pd.DataFrame([{"Clave": k, "Valor": v} for k, v in c.items() if k != "desglose_segundos"])
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(titulo)}</title>
<style>
.viz-root {{ color-scheme: light; --surface-1:#fcfcfb; --surface-2:#f2f1ed; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --text-muted:#7a7972; --grid:#e4e3de; --borde:#dddcd6;
  {''.join(f'--series-{i + 1}:{h};' for i, h in enumerate(SERIES))} --otras:{GRIS}; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232321; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#9a998f;
  --grid:#30302d; --borde:#3a3a37; {''.join(f'--series-{i + 1}:{h};' for i, h in enumerate(SERIES_OSC))} }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
.viz-root {{ background:var(--surface-1); color:var(--text-primary); min-height:100vh; padding:28px 32px 60px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 6px; }}
.sub {{ color:var(--text-secondary); margin:0 0 18px; }} .nota {{ color:var(--text-muted); font-size:13px; margin:4px 0 10px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
.kpi {{ background:var(--surface-2); border-radius:10px; padding:12px 14px; }}
.kpi .v {{ font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }} .kpi .k {{ color:var(--text-secondary); font-size:13px; }}
.graf {{ width:100%; height:420px; }} .graf.red {{ height:520px; }} .graf.chico {{ height:300px; }}
.tabla {{ overflow-x:auto; border:1px solid var(--borde); border-radius:10px; margin-top:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--borde); vertical-align:top; }}
th {{ color:var(--text-secondary); font-weight:600; background:var(--surface-2); position:sticky; top:0; }}
td {{ font-variant-numeric:tabular-nums; }}
</style>{plotly_js()}</head>
<body><div class="viz-root">
<h1>{_esc(titulo)}</h1>
<p class="sub">Período {_esc(d['periodo'])} · qué circula, desde cuándo y quién lo empuja</p>
<div class="kpis">{tiles}</div>

<h2>1 · Qué circula y desde cuándo</h2>
<p class="nota">Comentarios por narrativa cada 6 horas (las 8 más grandes; el resto en «Otras»). Las líneas punteadas marcan el origen estimado de cada narrativa emergente que se muestra (en el color de su serie las del top; en gris el resto).</p>
<div id="g_tiempo" class="graf"></div>
{_tabla(d['narr'])}
{f'<p class="nota">{d["ocultas"]["n"]} narrativas chicas ({d["ocultas"]["comentarios"]:,} comentarios) no se muestran</p>'.replace(",", ".") if d["ocultas"]["n"] else ''}

<h2>2 · Quién lo empuja: operaciones coordinadas</h2>
<p class="nota">Cada punto es una cuenta coordinada (tamaño = actividad); las líneas unen cuentas que actúan juntas una y otra vez, por encima de lo esperable por azar. Se destacan los 3 grupos más grandes.</p>
<div id="g_red" class="graf red"></div>
{_tabla(d['grupos']) if len(d['grupos']) else '<p class="nota">No se detectaron operaciones coordinadas.</p>'}

<h2>3 · Costo y escala</h2>
<div id="g_costo" class="graf chico"></div>
{_tabla(costo_tab)}
</div>
<script>
const D = {payload};
const css = n => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(n).trim();
const base = () => ({{ paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{ family:'system-ui, sans-serif', size:12, color:css('--text-secondary') }},
  margin:{{ l:56, r:16, t:10, b:46 }}, hoverlabel:{{ bgcolor:css('--surface-2'), font:{{ color:css('--text-primary') }} }},
  xaxis:{{ gridcolor:css('--grid'), zeroline:false, linecolor:css('--grid') }},
  yaxis:{{ gridcolor:css('--grid'), zeroline:false, rangemode:'tozero' }} }});
const color = (i, id) => id === 'Otras' ? css('--otras') : css('--series-' + (i + 1));
function dibujar() {{
  const sup = css('--surface-1');
  const tr = D.series.map((s, i) => ({{ x:D.x, y:s.y, name:s.id + ' · ' + s.titulo.slice(0, 48), type:'scatter', mode:'lines',
    stackgroup:'uno', line:{{ width:1.5, color:sup }}, fillcolor:color(i, s.id),
    hovertemplate:'<b>' + s.id + '</b> ' + s.titulo.replace(/</g, '&lt;') + '<br>%{{x}}: %{{y}} comentarios<extra></extra>' }}));
  const L1 = base();
  L1.yaxis.title = {{ text:'comentarios cada 6 h' }}; L1.hovermode = 'x unified'; L1.showlegend = true;
  L1.legend = {{ orientation:'h', y:-0.18, font:{{ color:css('--text-primary') }} }}; L1.margin.b = 110;
  L1.margin.t = 40;  /* las anotaciones de origen (y=1.0, yanchor bottom) viven en el margen superior */
  L1.shapes = D.origenes.map(o => ({{ type:'line', x0:o.x, x1:o.x, yref:'paper', y0:0, y1:1,
    line:{{ dash:'dash', width:2.5, color: o.i >= 0 ? css('--series-' + (o.i + 1)) : css('--otras') }} }}));
  L1.annotations = D.origenes.map((o, k) => ({{ x:o.x, yref:'paper', y:1.0 - 0.08 * (k % 7),
    text:'origen ' + o.id + ' · ' + o.titulo, showarrow:false, yanchor:'bottom',
    font:{{ size:11, color:css('--text-secondary') }} }}));
  Plotly.react('g_tiempo', tr, L1, {{ displaylogo:false, responsive:true }});

  const grupos = [...new Set(D.nodos.map(n => n.g))];
  const ex = [], ey = [];
  D.aristas.forEach(a => {{ ex.push(a[0], a[2], null); ey.push(a[1], a[3], null); }});
  const tr2 = [{{ x:ex, y:ey, mode:'lines', type:'scatter', line:{{ width:0.6, color:css('--grid') }}, hoverinfo:'skip', showlegend:false }}];
  grupos.forEach((g, i) => {{
    const ns = D.nodos.filter(n => n.g === g);
    tr2.push({{ x:ns.map(n => n.x), y:ns.map(n => n.y), mode:'markers', type:'scatter',
      name: i < 3 ? g : (i === 3 ? 'otros grupos' : g), legendgroup: i < 3 ? g : 'otros', showlegend: i <= 3,
      marker:{{ size:ns.map(n => 8 + Math.min(14, Math.sqrt(n.n) * 1.5)), color: i < 3 ? css('--series-' + (i + 1)) : css('--otras'),
        line:{{ width:2, color:sup }} }},
      text:ns.map(n => n.id + ' · ' + n.g + ' · ' + n.n + ' comentarios'), hovertemplate:'%{{text}}<extra></extra>' }});
  }});
  const L2 = base();
  L2.xaxis = {{ visible:false }}; L2.yaxis = {{ visible:false, scaleanchor:'x' }};
  L2.legend = {{ orientation:'h', y:-0.05, font:{{ color:css('--text-primary') }} }};
  Plotly.react('g_red', tr2, L2, {{ displaylogo:false, responsive:true }});

  const L3 = base();
  L3.margin.l = 120; L3.xaxis.title = {{ text:'segundos' }}; L3.yaxis.autorange = 'reversed'; L3.showlegend = false;
  Plotly.react('g_costo', [{{ x:D.etapas.seg, y:D.etapas.etapa, type:'bar', orientation:'h',
    marker:{{ color:css('--series-1') }}, hovertemplate:'%{{y}}: %{{x:.1f}} s<extra></extra>' }}], L3,
    {{ displaylogo:false, responsive:true }});
}}
dibujar();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', dibujar);
</script></body></html>"""


def correr(ctx) -> None:
    if not obtener(ctx.cfg, "reporte.activo", True):
        return
    d = datos_reporte(ctx.db, ctx.costo.a_dict(), ctx.cfg)
    destino = Path(obtener(ctx.cfg, "reporte.dir", "reportes")) / f"{ctx.datos.name}.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html_reporte(d, f"QualyData · narrativas y coordinación · {ctx.datos.name}"), encoding="utf-8")
    log.info("e10: reporte -> %s", destino)
