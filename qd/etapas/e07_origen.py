"""e07 · fecha_origen robusta por narrativa predicha.

core = comentarios no heredados con conf >= mediana de su narrativa, ordenados por fecha.
Se busca el primer comentario del core que tenga >= origen.m comentarios del core en las
siguientes origen.ventana_h horas (confirmación de ráfaga) y se toma el k-ésimo a partir de ahí.
Si ninguno confirma, el k-ésimo del core. Es rango, no percentil: no se corre con el tamaño.

Escribe narr(narrativa, n, titulo, frases_clave, ts_origen).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from qd.config import obtener

log = logging.getLogger("qd")
SALIDAS = ["narr"]


def origen_robusto(ts: np.ndarray, k: int, m: int, ventana_s: int) -> int:
    ts = np.sort(ts)
    if len(ts) == 0:
        return -1
    fin = np.searchsorted(ts, ts + ventana_s, side="right")
    ok = np.flatnonzero(fin - np.arange(len(ts)) >= m)
    i = int(ok[0]) if len(ok) else 0
    return int(ts[min(i + k - 1, len(ts) - 1)])


def correr(ctx) -> None:
    asig = ctx.db.leer("asig")
    com = ctx.db.leer("com")[["comentario_id", "ts"]]
    df = asig[asig["narrativa"] != ""].merge(com, on="comentario_id")
    k = int(obtener(ctx.cfg, "origen.k", 1))
    m = int(obtener(ctx.cfg, "origen.m", 3))
    v = int(float(obtener(ctx.cfg, "origen.ventana_h", 6)) * 3600)

    filas = []
    for lab, g in df.groupby("narrativa"):
        core = g[~g["heredada"].astype(bool)]
        if len(core):
            core = core[core["conf"] >= core["conf"].median()]
        ts = (core if len(core) else g)["ts"].to_numpy(np.int64)
        filas.append({"narrativa": lab, "ts_origen": origen_robusto(ts, k, m, v)})
    orig = pd.DataFrame(filas, columns=["narrativa", "ts_origen"])

    base = ctx.db.leer("narr_base")
    narr = base.merge(orig, on="narrativa", how="inner")
    narr["n"] = narr["narrativa"].map(df["narrativa"].value_counts()).fillna(0).astype(int)
    ctx.db.guardar("narr", narr[["narrativa", "n", "titulo", "frases_clave", "ts_origen"]])
    log.info("e07: %d narrativas con origen (k=%d, m=%d, ventana=%dh)", len(narr), k, m, v // 3600)
