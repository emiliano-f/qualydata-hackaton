"""Perfil de coordinación contra la verdad: por qué se detecta (o no) cada campaña y qué son los falsos positivos.

Uso: python herramientas/perfil_coord.py --datos datos/practica
Por campaña real: actividad, fuentes, horas, hashtags, y cuántos de sus pares coinciden en >= 2 publicaciones
(con distintas ventanas). Por grupo predicho genuino (falso positivo): tamaño, fuente dominante y antigüedad.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qd.etapas.e08_coord import eventos, pares  # noqa: E402


def perfil(t: dict, verdad: Path) -> None:
    from qd.entrada import ubicar

    vk = pd.read_csv(ubicar(verdad, "cuentas.csv"), dtype=str, keep_default_na=False)
    camp = vk.set_index("cuenta_id")["campana_id"]
    com, pub, cta = t["com"], t["pub"], t["cta"].set_index("cuenta_id")
    com = com.merge(pub[["publicacion_id", "fuente_id"]], on="publicacion_id", how="left")
    com["campana"] = com["cuenta_id"].map(camp).fillna("")
    com["hora"] = ((com["ts"] - 3 * 3600) // 3600) % 24
    fuentes_gen = com.loc[com["campana"] == "", "fuente_id"].value_counts(normalize=True)

    print("## Campañas reales")
    for c, g in com[com["campana"] != ""].groupby("campana"):
        ctas = g["cuenta_id"].unique()
        print(f"\n### {c}: {len(ctas)} cuentas, {len(g)} comentarios, {g['publicacion_id'].nunique()} publicaciones")
        print("- fuentes:", (g["fuente_id"].value_counts(normalize=True).round(2).head(5)).to_dict(),
              "| genuinos:", fuentes_gen.round(2).head(5).to_dict())
        print("- horas (top):", g["hora"].value_counts().head(6).to_dict())
        tags = g["texto"].str.lower().str.findall(r"#\w{3,}").explode().dropna()
        if len(tags):
            uso = com.loc[com["texto"].str.lower().str.contains("|".join(map(lambda x: x.replace("#", r"\#"), tags.unique()[:5]))), "cuenta_id"].nunique()
            print(f"- hashtags: {tags.value_counts().head(5).to_dict()} (cuentas totales que los usan: {uso})")
        print("- ejemplos:", g["texto"].head(3).tolist())
        ev = eventos(g)
        a, _ = pd.factorize(ev["cuenta_id"])
        p, _ = pd.factorize(ev["publicacion_id"])
        o = np.lexsort((ev["ts"].to_numpy(), p))
        for d in (120, 300, 600, 1800):
            k, n, s, _ = pares(p[o], ev["ts"].to_numpy()[o], a[o], np.zeros(len(o), np.int64), d, 0.0, len(ctas))
            tot = len(ctas) * (len(ctas) - 1) // 2
            print(f"- Δ={d:>4}s: pares con >=1 co-evento {len(k)}/{tot}, con >=2: {int((n >= 2).sum())}, "
                  f">=3: {int((n >= 3).sum())}")
        if "coord_pares_todos" in t:
            cp = t["coord_pares_todos"]
            dentro = cp[cp["u"].isin(ctas) & cp["v"].isin(ctas)]
            print(f"- en coord_pares_todos (n_pub>=min): {len(dentro)} pares; lift p10/p50/p90 = "
                  f"{np.round(dentro['lift'].quantile([.1, .5, .9]).to_numpy(), 2) if len(dentro) else '-'}")

    if "grupos" in t:
        gr = t["grupos"][t["grupos"]["cuenta_id"] != ""].copy()
        gr["campana"] = gr["cuenta_id"].map(camp).fillna("")
        print("\n## Grupos predichos")
        for gid, g in gr.groupby("grupo"):
            cg = com[com["cuenta_id"].isin(g["cuenta_id"])]
            edad = pd.to_numeric(cta.reindex(g["cuenta_id"])["edad_dias"], errors="coerce").median()
            f = cg["fuente_id"].value_counts(normalize=True)
            print(f"- {gid}: {len(g)} cuentas | campañas {g['campana'].replace('', '(gen)').value_counts().to_dict()} | "
                  f"fuente top {f.index[0] if len(f) else '-'} ({f.iloc[0]:.0%}) | edad mediana {edad:.0f} d | "
                  f"coment/cta {len(cg) / len(g):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"))
    ap.add_argument("--work", type=Path, default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    from diagnostico import leer_tablas

    t = leer_tablas(a.work or Path("work") / a.datos.name, ["com", "pub", "cta", "grupos", "coord_pares_todos"])
    perfil(t, a.datos / "verdad")


if __name__ == "__main__":
    main()
