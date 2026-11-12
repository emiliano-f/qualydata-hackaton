"""e02 · Embeddings de textos únicos.

Motor principal: intfloat/multilingual-e5-small (prefijo 'query: ', normalizado), con caché
global en cache/emb/<modelo>/ (claves = h_texto) que sirve entre corridas y entre sets.
Plan B solo a propósito (--embed tfidf o embed.motor: tfidf): TF-IDF de caracteres (3-5) + SVD 256,
sin descargas. Si el motor pedido es e5 y no carga, la etapa corta con RuntimeError: el respaldo
no es automático para que dos corridas no salgan con embeddings distintos sin avisar.

Escribe: tabla textos_emb(h_texto, motor) y work/<set>/emb.npy (float32, mismo orden).
Si ya existen para el mismo motor y los mismos textos, los reutiliza (corrida en caliente).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qd import vectores
from qd.config import obtener

log = logging.getLogger("qd")
SALIDAS = ["textos_emb"]


def _tfidf(norms, semilla: int = 42) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(norms)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3 if n >= 1000 else 1,
                          max_features=200_000, sublinear_tf=True, dtype=np.float32)
    M = vec.fit_transform([t or "_" for t in norms])
    comp = max(2, min(256, M.shape[1] - 1, n - 1))
    Z = TruncatedSVD(n_components=comp, random_state=semilla).fit_transform(M)
    return vectores.normalizar_filas(Z)


def _e5(h: np.ndarray, textos, cfg: dict) -> np.ndarray:
    modelo = obtener(cfg, "embed.modelo", "intfloat/multilingual-e5-small")
    prefijo = obtener(cfg, "embed.prefijo", "query: ")
    cdir = Path(obtener(cfg, "embed.cache_dir", "cache/emb")) / modelo.replace("/", "__")
    cdir.mkdir(parents=True, exist_ok=True)
    f_cl, f_ve = cdir / "claves.npy", cdir / "vectores.npy"
    claves = np.load(f_cl) if f_cl.exists() else np.empty(0, np.int64)
    vect = np.load(f_ve) if f_ve.exists() else np.empty((0, 0), np.float16)

    faltan = ~np.isin(h, claves)
    log.info("e5: %d textos, %d en caché, %d por calcular", len(h), int((~faltan).sum()), int(faltan.sum()))
    if faltan.any():
        import torch
        from sentence_transformers import SentenceTransformer

        disp = "cuda" if torch.cuda.is_available() else "cpu"
        m = SentenceTransformer(modelo, device=disp)
        m.max_seq_length = int(obtener(cfg, "embed.max_seq_length", 96))
        lote = [prefijo + str(textos[i]).strip()[:500] for i in np.flatnonzero(faltan)]
        t = time.perf_counter()
        V = m.encode(lote, batch_size=int(obtener(cfg, "embed.batch", 256)), normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True).astype(np.float16)
        log.info("e5: %d embeddings en %.0fs (%s, %.0f textos/s)", len(lote), time.perf_counter() - t, disp,
                 len(lote) / max(time.perf_counter() - t, 1e-9))
        claves = np.concatenate([claves, h[faltan]])
        vect = V if vect.size == 0 else np.vstack([vect, V])
        np.save(f_cl, claves)
        np.save(f_ve, vect)
    idx = pd.Index(claves).get_indexer(h)
    return vectores.normalizar_filas(vect[idx].astype(np.float32))


def correr(ctx) -> None:
    com = ctx.db.leer("com")[["h_texto", "texto", "norm"]]
    rep = com.drop_duplicates("h_texto").sort_values("h_texto")
    h = rep["h_texto"].to_numpy(np.int64)
    motor = ctx.embed

    # Reutilización en caliente
    if ctx.db.existe("textos_emb") and vectores.ruta(ctx).exists():
        prev = ctx.db.leer("textos_emb")
        if len(prev) == len(h) and (prev["motor"] == motor).all() and np.array_equal(prev["h_texto"].to_numpy(), h):
            log.info("e02: embeddings %s reutilizados (%d textos)", motor, len(h))
            ctx.costo.nota(f"embeddings locales {obtener(ctx.cfg, 'embed.modelo')}" if motor == "e5"
                           else "embeddings tfidf de caracteres + SVD (plan B)")
            return

    X = None
    if motor == "e5":
        try:
            X = _e5(h, rep["texto"].tolist(), ctx.cfg)
            ctx.costo.nota(f"embeddings locales {obtener(ctx.cfg, 'embed.modelo')}")
        except Exception as e:  # noqa: BLE001 - sin modelo/red: cortar, no cambiar de motor a escondidas
            msg = "e5 no disponible: corré make instalar con internet o usá --embed tfidf a propósito"
            log.error("%s (%s)", msg, e)
            raise RuntimeError(msg) from e
    if X is None:
        t = time.perf_counter()
        X = _tfidf(rep["norm"].tolist(), int(obtener(ctx.cfg, "micro.semilla", 42)))
        log.info("tfidf+svd: %s en %.1fs", X.shape, time.perf_counter() - t)
        ctx.costo.nota("embeddings tfidf de caracteres + SVD (plan B)")

    np.save(vectores.ruta(ctx), X)
    ctx.db.guardar("textos_emb", pd.DataFrame({"h_texto": h, "motor": motor}))
