"""Acceso a los embeddings de textos únicos (work/<set>/emb.npy alineado con la tabla textos_emb)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def ruta(ctx):
    return ctx.work / "emb.npy"


def cargar(ctx) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (h_texto[int64], X[float32, L2-normalizado]) en el mismo orden."""
    h = ctx.db.leer("textos_emb")["h_texto"].to_numpy(np.int64)
    X = np.load(ruta(ctx))
    if len(h) != len(X):
        raise RuntimeError(f"emb.npy ({len(X)}) no coincide con textos_emb ({len(h)}): correr desde e02")
    return h, X


def normalizar_filas(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)
