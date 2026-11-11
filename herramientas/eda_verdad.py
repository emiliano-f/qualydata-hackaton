"""Análisis de la verdad del set de práctica (H1 del plan, §4.1).

Uso: python herramientas/eda_verdad.py --datos datos/practica [--salida salida/eda_verdad.md]
Imprime y guarda en markdown: narrativas, ruido, postura, emergentes, respuestas, campañas,
trampas virales y frases copiadas, más semillas de marcadores y valores para config.sanidad.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qd import entrada  # noqa: E402
from qd.texto import DESACUERDO, normalizar  # noqa: E402

LINEAS = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINEAS.append(s)


def tabla(df: pd.DataFrame, n: int = 40):
    p("```")
    p(df.head(n).to_string())
    p("```")


def leer_verdad(d: Path, nombre: str) -> pd.DataFrame:
    r = entrada.ubicar(d, nombre)
    return pd.read_csv(r, dtype=str, keep_default_na=False) if r else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"))
    ap.add_argument("--salida", type=Path, default=Path("salida/eda_verdad.md"))
    a = ap.parse_args()
    v = a.datos / "verdad"

    p(f"# EDA de la verdad · {a.datos}\n")
    p("## Archivos en verdad/")
    for f in sorted(v.iterdir()):
        cols = list(leer_verdad(v, f.name.replace(".gz", "")).columns) if f.suffix in (".csv", ".gz") else ""
        p(f"- {f.name}: {cols}")

    com = entrada.leer(a.datos, "comentarios")
    com["ts"] = entrada.a_epoch(com["fecha"])
    com["norm"] = com["texto"].map(normalizar)
    com["n_tok"] = com["norm"].str.split().str.len()
    vc = leer_verdad(v, "comentarios.csv")
    m = com.merge(vc, on="comentario_id", how="left")
    vn = leer_verdad(v, "narrativas.csv")
    vk = leer_verdad(v, "cuentas.csv")
    cta = entrada.leer(a.datos, "cuentas")

    # --- Narrativas
    es_ruido = m["narrativa_id"] == "ninguna"
    p(f"\n## Narrativas\n- comentarios: {len(m)} | ruido: {100 * es_ruido.mean():.1f}%")
    nar = m[~es_ruido]
    t = nar.groupby("narrativa_id").agg(n=("comentario_id", "size"),
                                        pct_rechaza=("postura", lambda s: round(100 * (s == "rechaza").mean(), 1)),
                                        ts_min=("ts", "min"), ts_max=("ts", "max"))
    t["pct_total"] = (100 * t["n"] / len(m)).round(2)
    if len(vn):
        vn_i = vn.set_index("narrativa_id")
        t["emergente"] = vn_i.reindex(t.index)["emergente"]
        fo = entrada.a_epoch(vn_i.reindex(t.index)["fecha_origen"])
        t["origen_vs_min_h"] = ((t["ts_min"].astype("float") - fo.astype("float").to_numpy()) / 3600).round(2)
    t["dia_ini"] = ((t["ts_min"] - com["ts"].min()) / 86400).round(1)
    t["dia_fin"] = ((t["ts_max"] - com["ts"].min()) / 86400).round(1)
    tabla(t.drop(columns=["ts_min", "ts_max"]).sort_values("n", ascending=False))
    p(f"- narrativas: {len(t)} | postura global rechaza: {100 * (nar['postura'] == 'rechaza').mean():.1f}%")
    p("- origen_vs_min_h = (primer comentario de la verdad − fecha_origen real) en horas. Si es ~0, origen.k=1 es lo correcto.")
    if len(vn) and "descripcion" in vn.columns:
        p("\n### Descripciones")
        for r in vn.itertuples():
            p(f"- {r.narrativa_id} ({r.emergente}): {r.descripcion}")

    # --- Léxico actual
    rx = "|".join(rf"\b{x}\b" for x in DESACUERDO)
    tiene = nar["norm"].str.contains(rx)
    tp = (tiene & (nar["postura"] == "rechaza")).sum()
    p(f"\n## Léxico DESACUERDO actual\n- cobertura de rechaza: {100 * tp / max((nar['postura'] == 'rechaza').sum(), 1):.1f}%"
      f" | precisión: {100 * tp / max(tiene.sum(), 1):.1f}%")
    # n-gramas que discriminan rechaza vs apoya y aparecen en >=3 narrativas (no dependen del tema)
    frec = {"rechaza": Counter(), "apoya": Counter()}
    narrs = {"rechaza": {}, "apoya": {}}
    tot = {}
    for cls in frec:
        sub = nar[(nar["postura"] == cls) & (nar["respuesta_a"] == "")]
        sub = sub.sample(min(len(sub), 40000), random_state=0)
        tot[cls] = max(len(sub), 1)
        for nid, s_ in zip(sub["narrativa_id"], sub["norm"]):
            w = s_.split()
            grams = {" ".join(w[i:i + k]) for k in (1, 2, 3) for i in range(len(w) - k + 1)}
            frec[cls].update(grams)
            for g_ in grams:
                narrs[cls].setdefault(g_, set()).add(nid)
    for cls, otra in (("rechaza", "apoya"), ("apoya", "rechaza")):
        cand = []
        for g_, c in frec[cls].items():
            if c < 30 or len(narrs[cls].get(g_, ())) < 3:
                continue
            fr, fo = c / tot[cls], frec[otra].get(g_, 0) / tot[otra]
            cand.append((np.log((fr + 1e-4) / (fo + 1e-4)), g_, fr))
        cand.sort(reverse=True)
        p(f"- marcadores de '{cls}' (log-odds vs '{otra}', >=3 narrativas): "
          + ", ".join(f"{g_} ({100 * fr:.1f}%)" for _, g_, fr in cand[:35]))

    # --- Respuestas
    resp = m[m["respuesta_a"] != ""].merge(
        m[["comentario_id", "narrativa_id", "postura"]].rename(
            columns={"comentario_id": "respuesta_a", "narrativa_id": "n_p", "postura": "p_p"}), on="respuesta_a")
    p(f"\n## Respuestas\n- respuestas: {100 * (m['respuesta_a'] != '').mean():.1f}% de los comentarios")
    if len(resp):
        misma = resp["narrativa_id"] == resp["n_p"]
        p(f"- misma narrativa que el padre: {100 * misma.mean():.1f}% | padre ruido: {100 * (resp['n_p'] == 'ninguna').mean():.1f}%")
        r2 = resp[misma & (resp["narrativa_id"] != "ninguna")]
        igual = r2["postura"] == r2["p_p"]
        p(f"- con misma narrativa: postura igual {100 * igual.mean():.1f}% | opuesta {100 * (~igual).mean():.1f}%")
        for lim in (2, 3, 5, 8):
            c = r2[r2["n_tok"] <= lim]
            p(f"- cortas (<= {lim} tokens): {len(c)} | igual {100 * (c['postura'] == c['p_p']).mean():.1f}%")
        cortas = r2[r2["n_tok"] <= 6]
        for nombre, sel in (("ACUERDO (igual)", cortas["postura"] == cortas["p_p"]),
                            ("DESACUERDO (opuesta)", cortas["postura"] != cortas["p_p"])):
            p(f"- textos cortos {nombre}: " + ", ".join(f"{x} ({c})" for x, c in
                                                        Counter(cortas.loc[sel, "norm"]).most_common(25)))

    # --- Campañas
    if len(vk):
        k = vk.merge(cta, on="cuenta_id", how="left")
        k["ts_cre"] = entrada.a_epoch(k["fecha_creacion"])
        coord = k[k["campana_id"] != ""]
        p(f"\n## Campañas\n- cuentas: {len(k)} | coordinadas: {len(coord)} ({100 * len(coord) / len(k):.2f}%) |"
          f" campañas: {coord['campana_id'].nunique()}")
        mc = m.merge(vk, on="cuenta_id", how="left")
        p(f"- comentarios de cuentas coordinadas: {100 * (mc['campana_id'] != '').mean():.1f}%")
        filas = []
        t0 = com["ts"].min()
        gen = k[k["campana_id"] == ""]
        for cid, g in mc[mc["campana_id"] != ""].groupby("campana_id"):
            ctas = coord[coord["campana_id"] == cid]
            textos = g.groupby("norm")["cuenta_id"].nunique()
            dup = g["norm"].map(textos).ge(2).mean()
            gg = g.sort_values("ts")
            gap = gg.groupby("publicacion_id")["ts"].diff().median()
            cre = ctas["ts_cre"].astype("float")
            filas.append({"campana": cid, "cuentas": len(ctas), "coment": len(g),
                          "coment_x_cta": round(len(g) / len(ctas), 1),
                          "pubs_x_cta": round(g.groupby("cuenta_id")["publicacion_id"].nunique().mean(), 1),
                          "pct_texto_dup": round(100 * dup, 1), "gap_med_s": gap,
                          "creacion_std_dias": round(cre.std() / 86400, 1),
                          "edad_med_dias": round((t0 - cre.median()) / 86400, 0),
                          "seguidores_med": pd.to_numeric(ctas["seguidores"], errors="coerce").median(),
                          "narrativas": g["narrativa_id"].nunique(),
                          "postura_rech_pct": round(100 * (g["postura"] == "rechaza").mean(), 1)})
        tabla(pd.DataFrame(filas))
        p(f"- genuinas: edad mediana {round((t0 - gen['ts_cre'].astype('float').median()) / 86400)} días, "
          f"seguidores mediana {pd.to_numeric(gen['seguidores'], errors='coerce').median()}")
        if "descripcion" in vk.columns:
            pass
        camp = leer_verdad(v, "campanas.csv")
        if len(camp):
            p("\n### Descripción de campañas")
            tabla(camp)

        # --- Trampas
        p("\n## Trampas")
        m10 = mc.assign(b=(mc["ts"] // 600))
        pico = m10.groupby(["publicacion_id", "b"]).agg(n=("comentario_id", "size"),
                                                        pct_coord=("campana_id", lambda s: round(100 * (s != "").mean(), 1)))
        p("- picos de 10 minutos por publicación (top 8):")
        tabla(pico.sort_values("n", ascending=False).head(8))
        larg = mc[mc["n_tok"] >= 6]
        frases = larg.groupby("norm").agg(ctas=("cuenta_id", "nunique"),
                                          pct_coord=("campana_id", lambda s: round(100 * (s != "").mean(), 1)))
        p("- frases largas usadas por más cuentas (top 10):")
        tabla(frases.sort_values("ctas", ascending=False).head(10))

        p("\n## Para config.yaml › sanidad")
        p(f"  pct_ruido: {100 * es_ruido.mean():.1f}\n  n_narrativas: {len(t)}\n  pct_coord: {100 * len(coord) / len(k):.2f}")

    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text("\n".join(LINEAS), encoding="utf-8")
    print(f"\n→ {a.salida}")


if __name__ == "__main__":
    main()
