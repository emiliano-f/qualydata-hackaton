"""e03 · Micro-clusters: MiniBatchKMeans sobreparticionado, ajustado sobre una muestra
estratificada por día de aparición (así las narrativas que nacen tarde tienen peso propio)
y aplicado a todos los textos únicos.

Escribe: micro_txt(h_texto, micro, conf) y work/<set>/centroides.npy (L2-normalizados).
conf = coseno del texto con su centroide.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from qd import vectores
from qd.config import obtener

log = logging.getLogger("qd")
SALIDAS = ["micro_txt"]


def muestra_por_dia(dia: np.ndarray, por_dia: int, rng) -> np.ndarray:
    idx = []
    for d in np.unique(dia):
        cand = np.flatnonzero(dia == d)
        idx.append(cand if len(cand) <= por_dia else rng.choice(cand, por_dia, replace=False))
    return np.sort(np.concatenate(idx))


def elegir_k(n: int, cfg: dict, n_muestra: int) -> int:
    k = round(obtener(cfg, "micro.k_factor", 1.0) * np.sqrt(n) / 2)
    k = int(np.clip(k, obtener(cfg, "micro.k_min", 80), obtener(cfg, "micro.k_max", 250)))
    return max(2, min(k, n_muestra // 3 if n_muestra >= 6 else n_muestra))


def correr(ctx) -> None:
    from sklearn.cluster import MiniBatchKMeans

    h, X = vectores.cargar(ctx)
    tx = ctx.db.leer("textos").set_index("h_texto").loc[h]
    semilla = int(obtener(ctx.cfg, "micro.semilla", 42))
    rng = np.random.default_rng(semilla)

    s = muestra_por_dia(tx["dia_min"].to_numpy(), int(obtener(ctx.cfg, "micro.por_dia", 3000)), rng)
    k = elegir_k(len(h), ctx.cfg, len(s))
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=semilla)
    km.fit(X[s], sample_weight=np.log1p(tx["n"].to_numpy()[s]).astype(np.float64))
    lab = km.predict(X)
    C = vectores.normalizar_filas(km.cluster_centers_)
    conf = (X * C[lab]).sum(axis=1)

    np.save(ctx.work / "centroides.npy", C)
    ctx.db.guardar("micro_txt", pd.DataFrame({"h_texto": h, "micro": lab.astype(np.int64),
                                              "conf": conf.astype(np.float64)}))
    log.info("e03: %d textos (muestra %d), k=%d, micro-clusters usados=%d, conf mediana=%.3f",
             len(h), len(s), k, len(np.unique(lab)), float(np.median(conf)))
