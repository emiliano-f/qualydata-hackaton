"""Arma un "pseudo-final" a partir de práctica para medir generalización: primeros N días y una fracción
de las publicaciones (el set final trae ~150.000 comentarios en 10 días). Filtra también la verdad.

Uso: python herramientas/pseudo_final.py --origen datos/practica --destino work/datos/p10 [--dias 10] [--frac 0.75]
     make score SET=p10 DATOS=work/datos/p10 SALIDA=salida/p10
Si el puntaje cae mucho respecto de práctica completa, hay un parámetro que no generaliza.
"""
import argparse
import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def leer(p: Path) -> pd.DataFrame:
    for c in (p, p.with_name(p.name + ".gz")):
        if c.exists():
            return pd.read_csv(c, dtype=str, keep_default_na=False)
    raise FileNotFoundError(p)


def escribir(df: pd.DataFrame, p: Path, gz: bool):
    if gz:
        with gzip.open(p.with_name(p.name + ".gz"), "wt", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
    else:
        df.to_csv(p, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", type=Path, default=Path("datos/practica"))
    ap.add_argument("--destino", type=Path, default=Path("work/datos/p10"))
    ap.add_argument("--dias", type=float, default=10)
    ap.add_argument("--frac", type=float, default=0.75)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    o, d = a.origen, a.destino
    (d / "verdad").mkdir(parents=True, exist_ok=True)

    com = leer(o / "comentarios.csv")
    ts = pd.to_datetime(com["fecha"], utc=True, errors="coerce", format="ISO8601")
    corte = ts.min() + pd.Timedelta(days=a.dias)
    pub = leer(o / "publicaciones.csv")
    rng = np.random.default_rng(a.semilla)
    pubs = pub["publicacion_id"].to_numpy()
    elegidas = set(rng.choice(pubs, size=int(len(pubs) * a.frac), replace=False))
    com = com[(ts < corte).to_numpy() & com["publicacion_id"].isin(elegidas).to_numpy()]
    ids = set(com["comentario_id"])
    com.loc[~com["respuesta_a"].isin(ids), "respuesta_a"] = ""
    ctas = set(com["cuenta_id"])

    escribir(com, d / "comentarios.csv", True)
    escribir(pub[pub["publicacion_id"].isin(set(com["publicacion_id"]))], d / "publicaciones.csv", True)
    cu = leer(o / "cuentas.csv")
    escribir(cu[cu["cuenta_id"].isin(ctas)], d / "cuentas.csv", True)
    shutil.copy(o / "fuentes.csv", d / "fuentes.csv")

    v = o / "verdad"
    vc = leer(v / "comentarios.csv")
    escribir(vc[vc["comentario_id"].isin(ids)], d / "verdad" / "comentarios.csv", False)
    vk = leer(v / "cuentas.csv")
    escribir(vk[vk["cuenta_id"].isin(ctas)], d / "verdad" / "cuentas.csv", False)
    vn = leer(v / "narrativas.csv")
    presentes = set(vc.loc[vc["comentario_id"].isin(ids), "narrativa_id"])
    escribir(vn[vn["narrativa_id"].isin(presentes)], d / "verdad" / "narrativas.csv", False)
    print(f"pseudo-final: {len(com)} comentarios, {len(ctas)} cuentas, {len(presentes - {'ninguna'})} narrativas "
          f"({a.dias:g} días, {a.frac:.0%} de publicaciones) -> {d}")


if __name__ == "__main__":
    main()
