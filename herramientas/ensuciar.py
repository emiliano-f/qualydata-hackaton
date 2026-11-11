"""Copia un set y lo ensucia a propósito para probar la robustez del pipeline.

Inyecta: textos vacíos, fechas inválidas y en otro formato, IDs duplicados, respuesta_a a
comentarios inexistentes, emojis y caracteres de control, me_gusta no numérico, una línea
con columnas de más, BOM, una cuenta sin metadatos y una columna faltante en cuentas.
"""
import argparse
import gzip
import random
import shutil
from pathlib import Path

import pandas as pd


def leer(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", type=Path, required=True)
    ap.add_argument("--destino", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.005)
    ap.add_argument("--semilla", type=int, default=13)
    a = ap.parse_args()
    rnd = random.Random(a.semilla)
    a.destino.mkdir(parents=True, exist_ok=True)
    for nombre in ("fuentes.csv", "publicaciones.csv.gz"):
        if (a.origen / nombre).exists():
            shutil.copy(a.origen / nombre, a.destino / nombre)

    com = leer(a.origen / "comentarios.csv.gz")
    n = len(com)
    idx = lambda: rnd.sample(range(n), max(1, int(n * a.frac)))  # noqa: E731
    com.loc[idx(), "texto"] = ""
    com.loc[idx(), "fecha"] = "no-es-fecha"
    com.loc[idx(), "fecha"] = "2026-09-03 10:00:00Z"  # formato distinto pero válido
    com.loc[idx(), "respuesta_a"] = "c_que_no_existe"
    com.loc[idx(), "me_gusta"] = "muchos"
    i = idx()
    com.loc[i, "texto"] = com.loc[i, "texto"] + " 🔥🔥\x07\t​ ¡¡NO!!"
    com = pd.concat([com, com.sample(max(1, int(n * a.frac)), random_state=a.semilla)], ignore_index=True)
    with gzip.open(a.destino / "comentarios.csv.gz", "wt", encoding="utf-8-sig", newline="") as f:
        com.to_csv(f, index=False)
        f.write("linea_rota,p001,u0001,2026-09-02T10:00:00-03:00,texto,,3,COLUMNA_DE_MAS,OTRA\n")

    cta = leer(a.origen / "cuentas.csv.gz")
    cta = cta.drop(columns=["siguiendo"]).iloc[1:]  # falta una columna y una cuenta
    cta.loc[cta.sample(5, random_state=1).index, "fecha_creacion"] = ""
    with gzip.open(a.destino / "cuentas.csv.gz", "wt", encoding="utf-8", newline="") as f:
        cta.to_csv(f, index=False)
    print(f"sucio: {len(com)} filas -> {a.destino}")


if __name__ == "__main__":
    main()
