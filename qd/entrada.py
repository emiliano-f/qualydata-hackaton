"""Lectura tolerante de los CSV de entrada. La comparten la ingesta y el validador,
así ambos ven exactamente el mismo universo de IDs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

log = logging.getLogger("qd")

ARCHIVOS = {
    "fuentes": ("fuentes.csv", ["fuente_id", "nombre", "tipo"]),
    "publicaciones": ("publicaciones.csv", ["publicacion_id", "fuente_id", "fecha", "titular"]),
    "comentarios": ("comentarios.csv", ["comentario_id", "publicacion_id", "cuenta_id", "fecha", "texto",
                                        "respuesta_a", "me_gusta"]),
    "cuentas": ("cuentas.csv", ["cuenta_id", "usuario", "fecha_creacion", "seguidores", "siguiendo"]),
}
ID_COLS = {"fuente_id", "publicacion_id", "comentario_id", "cuenta_id", "respuesta_a"}


def ubicar(datos: Path, base: str) -> Optional[Path]:
    for cand in (datos / f"{base}.gz", datos / base):
        if cand.exists():
            return cand
    return None


def leer(datos: Path, clave: str, obligatorio: bool = True) -> pd.DataFrame:
    """Todo como texto, sin NaN (keep_default_na=False), líneas rotas salteadas, columnas faltantes en ''."""
    base, columnas = ARCHIVOS[clave]
    ruta = ubicar(datos, base)
    if ruta is None:
        if obligatorio:
            raise FileNotFoundError(f"Falta {datos / base}(.gz)")
        log.warning("Falta %s: sigo con tabla vacía", base)
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columnas})
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, na_filter=False, on_bad_lines="skip",
                     encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    faltan: List[str] = [c for c in columnas if c not in df.columns]
    if faltan:
        log.warning("%s: faltan columnas %s, se completan vacías", base, faltan)
        for c in faltan:
            df[c] = ""
    df = df[columnas].astype(object)
    for c in columnas:
        if c in ID_COLS:
            df[c] = df[c].astype(str).str.strip()
    return df


def ids_comentarios(datos: Path) -> pd.Series:
    """Universo de comentario_id que la entrega tiene que cubrir (únicos, no vacíos)."""
    df = leer(datos, "comentarios")
    s = df["comentario_id"]
    return pd.Series(s[s != ""].unique(), name="comentario_id")


def ids_cuentas(datos: Path) -> pd.Series:
    a = leer(datos, "cuentas", obligatorio=False)["cuenta_id"]
    b = leer(datos, "comentarios")["cuenta_id"]
    s = pd.concat([a, b], ignore_index=True)
    return pd.Series(s[s != ""].unique(), name="cuenta_id")


def a_epoch(serie: pd.Series) -> pd.Series:
    """ISO con zona -> segundos epoch (Int64 con NA). Independiente de la resolución de pandas (ns/us)."""
    s = serie.astype(object).where(serie.astype(str).str.strip() != "")
    dt = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    seg = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    return seg.astype("Int64")
