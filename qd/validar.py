"""Validador de entrega: falla (exit 1) ante cualquier cosa que haga perder puntos en silencio.

Uso: python -m qd.validar --datos datos/final --entrega entrega [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from qd import entrada

RE_FECHA = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d-03:00$")
BASURA = {"nan", "NaN", "None", "NaT", "<NA>"}
CLAVES_COSTO = ["comentarios_procesados", "segundos", "tokens_ia", "usd_ia", "usd_por_mil_comentarios",
                "hardware", "almacenamiento", "notas"]


def _leer(ruta: Path) -> Optional[pd.DataFrame]:
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, na_filter=False)


def validar(datos: Path, entrega: Path, sanidad: Optional[dict] = None) -> Tuple[List[str], List[str]]:
    err: List[str] = []
    av: List[str] = []
    c = _leer(entrega / "comentarios.csv")
    n = _leer(entrega / "narrativas.csv")
    k = _leer(entrega / "cuentas.csv")
    for nombre, df in [("comentarios.csv", c), ("narrativas.csv", n), ("cuentas.csv", k)]:
        if df is None:
            err.append(f"falta {nombre}")
    if not (entrega / "costo.json").exists():
        err.append("falta costo.json")
    if err:
        return err, av

    # 1. columnas
    for nombre, df, cols in [("comentarios.csv", c, ["comentario_id", "narrativa", "postura"]),
                             ("narrativas.csv", n, ["narrativa", "titulo", "frases_clave", "fecha_origen"]),
                             ("cuentas.csv", k, ["cuenta_id", "grupo"])]:
        if list(df.columns) != cols:
            err.append(f"{nombre}: columnas {list(df.columns)} != {cols}")
    if err:
        return err, av

    # 2. universo de comentarios
    esperados = set(entrada.ids_comentarios(datos))
    vistos = c["comentario_id"]
    if vistos.duplicated().any():
        err.append(f"comentarios.csv: {int(vistos.duplicated().sum())} IDs duplicados")
    faltan, sobran = esperados - set(vistos), set(vistos) - esperados
    if faltan:
        err.append(f"comentarios.csv: faltan {len(faltan)} IDs (ej. {sorted(faltan)[:3]})")
    if sobran:
        err.append(f"comentarios.csv: sobran {len(sobran)} IDs (ej. {sorted(sobran)[:3]})")

    # 3. postura completa
    malas = ~c["postura"].isin(["apoya", "rechaza"])
    if malas.any():
        err.append(f"comentarios.csv: {int(malas.sum())} posturas fuera de {{apoya, rechaza}}")

    # 4. basura textual
    for nombre, df in [("comentarios.csv", c), ("narrativas.csv", n), ("cuentas.csv", k)]:
        sucias = df.isin(BASURA).sum()
        for col, cant in sucias[sucias > 0].items():
            err.append(f"{nombre}.{col}: {int(cant)} celdas con nan/None/NaT")

    # 5. etiquetas ⊂ narrativas
    usadas = set(c.loc[~c["narrativa"].str.lower().isin(["", "ninguna"]), "narrativa"])
    sin_fila = usadas - set(n["narrativa"])
    if sin_fila:
        err.append(f"{len(sin_fila)} etiquetas de comentarios.csv sin fila en narrativas.csv (origen = 0)")
    if n["narrativa"].duplicated().any():
        err.append("narrativas.csv: etiquetas duplicadas")

    # 6. fechas
    for lab, f in zip(n["narrativa"], n["fecha_origen"]):
        ok = bool(RE_FECHA.match(f))
        if ok:
            try:
                ok = datetime.fromisoformat(f).tzinfo is not None
            except ValueError:
                ok = False
        if not ok:
            err.append(f"narrativas.csv: fecha_origen inválida en {lab!r}: {f!r}")

    # 7. universo de cuentas
    faltan_c = set(entrada.ids_cuentas(datos)) - set(k["cuenta_id"])
    if faltan_c:
        av.append(f"cuentas.csv: faltan {len(faltan_c)} cuentas (cuentan como genuinas)")
    if k["cuenta_id"].duplicated().any():
        err.append("cuentas.csv: cuenta_id duplicados")

    # 8. costo.json
    try:
        costo = json.loads((entrega / "costo.json").read_text(encoding="utf-8"))
        for clave in CLAVES_COSTO:
            if clave not in costo:
                err.append(f"costo.json: falta la clave {clave}")
    except json.JSONDecodeError as e:
        err.append(f"costo.json inválido: {e}")

    # 9. sanidad (solo avisos), rangos [0.5x, 2x] de lo observado en práctica
    obs = {"pct_ruido": 100 * c["narrativa"].isin(["", "ninguna"]).mean(),
           "n_narrativas": float(len(n)),
           "pct_coord": 100 * (k["grupo"] != "").mean() if len(k) else 0.0}
    for clave, valor in obs.items():
        ref = (sanidad or {}).get(clave)
        if ref and not (0.5 * ref <= valor <= 2 * ref):
            av.append(f"sanidad: {clave}={valor:.1f} fuera de [{0.5 * ref:.1f}, {2 * ref:.1f}] (práctica={ref})")
    return err, av


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, required=True)
    ap.add_argument("--entrega", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    a = ap.parse_args()
    sanidad = {}
    if a.config.exists():
        from qd.config import cargar

        sanidad = cargar(a.config).get("sanidad", {}) or {}
    err, av = validar(a.datos, a.entrega, sanidad)
    for x in av:
        print(f"AVISO  {x}")
    for x in err:
        print(f"ERROR  {x}")
    print("VALIDACIÓN OK" if not err else f"VALIDACIÓN FALLÓ ({len(err)} errores)")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
