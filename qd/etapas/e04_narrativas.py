"""e04 · Narrativas: consolidación de micro-clusters (sin LLM en v0).

1. Aglomerativo (coseno, average) sobre los centroides usados, con umbral RELATIVO:
   T = cuantil q_T de la distancia de cada centroide a su vecino más cercano.
2. Afinidad por respuestas: une narrativas cuando muchas respuestas cruzan de una a otra
   (apoyo y rechazo de la misma afirmación suelen quedar en micro-clusters distintos).
3. Ruido v0 (lo reemplaza e05): textos de <= ruido.max_tok tokens que no son respuesta.

Escribe: asig_narr(comentario_id, micro, narrativa, conf, ruido_v0)
         micro_narr(micro, narrativa, n_com)
         narr_base(narrativa, n, titulo, frases_clave)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from qd.config import obtener
from qd.texto import STOPWORDS

log = logging.getLogger("qd")
SALIDAS = ["asig_narr", "micro_narr", "narr_base"]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def umbral_relativo(D: np.ndarray, q: float) -> float:
    Dn = D.copy()
    np.fill_diagonal(Dn, np.inf)
    return float(np.quantile(Dn.min(axis=1), q))


def consolidar(C: np.ndarray, q_T: float):
    """Devuelve (grupo por centroide, matriz de distancias, T)."""
    from sklearn.cluster import AgglomerativeClustering

    D = np.clip(1.0 - C @ C.T, 0.0, 2.0)
    if len(C) < 2:
        return np.zeros(len(C), dtype=int), D, 0.0
    T = umbral_relativo(D, q_T)
    agg = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                  distance_threshold=max(T, 1e-6)).fit(C)
    return agg.labels_, D, T


def unir_por_respuestas(grupo, D, T, micro_ids, com_m, r_min, delta):
    """com_m: comentario_id, micro, respuesta_a. Une grupos con mucha respuesta cruzada."""
    pos = {m: i for i, m in enumerate(micro_ids)}
    par = com_m[["comentario_id", "micro"]].rename(columns={"comentario_id": "respuesta_a", "micro": "micro_p"})
    r = com_m[com_m["respuesta_a"] != ""].merge(par, on="respuesta_a")
    r = r[r["micro"] != r["micro_p"]]
    if r.empty:
        return grupo, 0
    n_micro = com_m.groupby("micro").size()
    cnt = r.groupby(["micro", "micro_p"]).size().reset_index(name="c")
    cnt["a"] = np.minimum(cnt["micro"], cnt["micro_p"])
    cnt["b"] = np.maximum(cnt["micro"], cnt["micro_p"])
    cnt = cnt.groupby(["a", "b"])["c"].sum().reset_index()
    cnt["R"] = cnt["c"] / np.minimum(n_micro.loc[cnt["a"]].to_numpy(), n_micro.loc[cnt["b"]].to_numpy())
    uf = UnionFind(int(grupo.max()) + 1)
    uniones = 0
    for a, b, R in cnt[["a", "b", "R"]].itertuples(index=False):
        i, j = pos[a], pos[b]
        if R >= r_min and D[i, j] <= T + delta and uf.find(grupo[i]) != uf.find(grupo[j]):
            uf.union(grupo[i], grupo[j])
            uniones += 1
    return np.array([uf.find(g) for g in grupo]), uniones


def terminos_clave(textos_por_narr: pd.Series, top: int = 6):
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(textos_por_narr) == 0:
        return {}
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORDS), sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z0-9]{3,}\b")
    try:
        M = vec.fit_transform(textos_por_narr.to_numpy())
    except ValueError:  # vocabulario vacío
        return {k: [] for k in textos_por_narr.index}
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for i, k in enumerate(textos_por_narr.index):
        fila = M.getrow(i).toarray().ravel()
        out[k] = [terms[j] for j in np.argsort(-fila)[:top] if fila[j] > 0]
    return out


def correr(ctx) -> None:
    C_all = np.load(ctx.work / "centroides.npy")
    mt = ctx.db.leer("micro_txt")
    com = ctx.db.leer("com")[["comentario_id", "h_texto", "respuesta_a", "n_tok", "norm"]]
    com = com.merge(mt, on="h_texto", how="left")

    micro_ids = np.sort(mt["micro"].unique())
    C = C_all[micro_ids]
    grupo, D, T = consolidar(C, float(obtener(ctx.cfg, "narr.q_T", 0.6)))
    n_aglo = len(np.unique(grupo))
    grupo, uniones = unir_por_respuestas(grupo, D, T, micro_ids, com[["comentario_id", "micro", "respuesta_a"]],
                                         float(obtener(ctx.cfg, "narr.r_resp", 0.15)),
                                         float(obtener(ctx.cfg, "narr.delta_gris", 0.06)))

    # Etiquetas n001.. ordenadas por cantidad de comentarios
    m2g = dict(zip(micro_ids, grupo))
    com["g"] = com["micro"].map(m2g)
    orden = com["g"].value_counts().index
    g2lab = {g: f"n{i + 1:03d}" for i, g in enumerate(orden)}
    com["narrativa"] = com["g"].map(g2lab)

    max_tok = int(obtener(ctx.cfg, "ruido.max_tok", 2))
    com["ruido_v0"] = (com["n_tok"] <= max_tok) & (com["respuesta_a"] == "")

    asig = com[["comentario_id", "micro", "narrativa", "conf", "ruido_v0"]].copy()
    micro_narr = (com.groupby("micro").agg(narrativa=("narrativa", "first"), n_com=("comentario_id", "size"))
                  .reset_index())

    con_texto = com[~com["ruido_v0"]]
    muestra = con_texto.sample(frac=1.0, random_state=0).groupby("narrativa").head(3000)
    docs = muestra.groupby("narrativa")["norm"].apply(" ".join)
    claves = terminos_clave(docs)
    n_por = com["narrativa"].value_counts()
    narr_base = pd.DataFrame([{"narrativa": lab, "n": int(n_por.get(lab, 0)),
                               "titulo": " / ".join(claves.get(lab, [])[:3]) or lab,
                               "frases_clave": "|".join(claves.get(lab, []))} for lab in g2lab.values()])

    ctx.db.guardar("asig_narr", asig)
    ctx.db.guardar("micro_narr", micro_narr)
    ctx.db.guardar("narr_base", narr_base)
    log.info("e04: %d micro -> %d por aglomerativo (T=%.4f) -> %d narrativas (+%d uniones por respuestas); "
             "ruido_v0=%.1f%%", len(micro_ids), n_aglo, T, len(g2lab), uniones, 100 * com["ruido_v0"].mean())
