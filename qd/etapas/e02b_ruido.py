"""e02b · Ruido por texto, antes de los micro-clusters.

Hecho de la verdad de práctica: el ruido nunca responde ni recibe respuestas. Por cada texto único:
  - si alguna aparición responde o fue respondida -> narrativa (decidido, certeza);
  - si nunca está involucrado en respuestas y aparece n >= n_min veces, con (1-p)^n_min <= ruido.alfa
    (p = tasa global de involucramiento) -> ruido (decidido);
  - el resto (textos raros) se decide por vecinos: voto de los ruido.knn_k textos decididos más
    parecidos (coseno, ponderado). Así entran las variantes con typos de las plantillas de ruido.
Las plantillas de ruido repetidas cientos de veces son pocos textos únicos: sin este paso KMeans
no les da centroide propio y contaminan los clusters de narrativas.

Escribe textos_ruido(h_texto, ruido_txt, decision, p_ruido).
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from qd import vectores
from qd.config import obtener

log = logging.getLogger("qd")
SALIDAS = ["textos_ruido"]


def votar_knn(Xu: np.ndarray, Xd: np.ndarray, yd: np.ndarray, k: int, bloque: int = 2048) -> np.ndarray:
    """Probabilidad de ruido de cada fila de Xu según sus k vecinos más parecidos en Xd."""
    out = np.zeros(len(Xu))
    k = max(1, min(k, len(Xd)))
    for i in range(0, len(Xu), bloque):
        S = Xu[i:i + bloque] @ Xd.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(S, idx, axis=1).clip(min=1e-6)
        out[i:i + bloque] = (w * yd[idx]).sum(1) / w.sum(1)
    return out


def correr(ctx) -> None:
    cfg = ctx.cfg
    h, X = vectores.cargar(ctx)
    com = ctx.db.leer("com")[["comentario_id", "h_texto", "respuesta_a"]]
    respondidos = set(com.loc[com["respuesta_a"] != "", "respuesta_a"])
    com["inv"] = (com["respuesta_a"] != "") | com["comentario_id"].isin(respondidos)
    g = com.groupby("h_texto").agg(n=("comentario_id", "size"), inv=("inv", "sum")).reindex(h).fillna(0)

    p = float(com["inv"].mean())
    alfa = float(obtener(cfg, "ruido.alfa", 0.01))
    if p <= 0 or p >= 1:
        log.warning("e02b: sin estructura de respuestas (p=%.3f): no se marca ruido por texto", p)
        ruido = np.zeros(len(h), dtype=bool)
        dec = np.full(len(h), "sin_datos", dtype=object)
        p_r = np.zeros(len(h))
    else:
        n_min = max(3, math.ceil(math.log(alfa) / math.log(1 - p)))
        es_narr = (g["inv"] > 0).to_numpy()
        es_ruido = ((g["inv"] == 0) & (g["n"] >= n_min)).to_numpy()
        indef = ~(es_narr | es_ruido)
        p_r = es_ruido.astype(float)
        dec = np.where(es_narr, "narrativa", np.where(es_ruido, "ruido", "knn")).astype(object)
        if indef.any() and es_ruido.any() and es_narr.any():
            d = es_narr | es_ruido
            p_r[indef] = votar_knn(X[indef], X[d], es_ruido[d].astype(float), int(obtener(cfg, "ruido.knn_k", 10)))
        umbral = float(obtener(cfg, "ruido.knn_umbral", 0.5))
        ruido = es_ruido | (indef & (p_r >= umbral))
        n_com = g["n"].to_numpy()
        log.info("e02b: p_involucrado=%.3f -> n_min=%d | textos: %d narrativa, %d ruido seguro, %d por vecinos "
                 "(%d ruido) | comentarios ruido=%.1f%%", p, n_min, int(es_narr.sum()), int(es_ruido.sum()),
                 int(indef.sum()), int((indef & ruido).sum()), 100 * n_com[ruido].sum() / max(n_com.sum(), 1))
    ctx.db.guardar("textos_ruido", pd.DataFrame({"h_texto": h, "ruido_txt": ruido, "decision": dec,
                                                 "p_ruido": p_r.astype(float)}))
