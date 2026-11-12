"""e04 · Narrativas: ruido estructural + consolidación de micro-clusters + fusión por respuestas.

Hechos de la verdad de práctica que se explotan (no dependen del tema):
  - las respuestas son SIEMPRE de la narrativa del padre y nunca son ruido;
  - nadie le responde al ruido.
Pasos:
  1. Ruido estructural: micro-clusters que casi no reciben ni hacen respuestas
     (score = %recibe + %es_respuesta <= ruido.q_rel * mediana). Además ruido v0: textos de
     <= ruido.max_tok tokens que no son respuesta. Nunca es ruido una respuesta ni un comentario respondido.
  2. Aglomerativo (coseno, average) sobre los centroides NO ruido, umbral relativo T = cuantil q_T
     de la distancia al vecino más cercano.
  3. Fusión por respuestas: Louvain (resolución narr.gamma_resp) sobre el grafo de grupos cuyas aristas
     son respuestas largas cruzadas (>= narr.min_resp).

Escribe: asig_narr(comentario_id, micro, narrativa, conf, es_ruido)
         micro_narr(micro, narrativa, n_com, ruido_micro, score_resp)
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
            return True
        return False


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


def estadisticas_respuesta(com: pd.DataFrame) -> pd.DataFrame:
    """Por micro: fracción de comentarios que son respuesta y fracción que reciben alguna respuesta."""
    respondidos = set(com.loc[com["respuesta_a"] != "", "respuesta_a"])
    c = com.assign(es_resp=com["respuesta_a"] != "", recibe=com["comentario_id"].isin(respondidos))
    st = c.groupby("micro").agg(n_com=("comentario_id", "size"), es_resp=("es_resp", "mean"),
                                recibe=("recibe", "mean"))
    st["score_resp"] = st["es_resp"] + st["recibe"]
    return st


def aristas_respuesta(com: pd.DataFrame, corta: int) -> pd.DataFrame:
    """Respuestas largas como aristas micro_hijo -> micro_padre (las cortas acompañan y no aportan bando)."""
    par = com[["comentario_id", "micro"]].rename(columns={"comentario_id": "respuesta_a", "micro": "micro_p"})
    r = com[(com["respuesta_a"] != "") & (com["n_tok"] > corta)][["micro", "respuesta_a"]].merge(par, on="respuesta_a")
    return r[["micro", "micro_p"]]


def fusionar_por_respuestas(grupo_de_micro: dict, aristas: pd.DataFrame, gamma: float, min_resp: int,
                            semilla: int = 42):
    """Louvain sobre el grafo de grupos con aristas = respuestas largas cruzadas (>= min_resp).
    Las respuestas son siempre de la narrativa del padre: comunidades densas = narrativas.
    Un micro-cluster mezclado queda en UNA comunidad en vez de encadenar dos narrativas."""
    import networkx as nx

    if aristas.empty:
        return grupo_de_micro, 0
    g_h = aristas["micro"].map(grupo_de_micro)
    g_p = aristas["micro_p"].map(grupo_de_micro)
    e = pd.DataFrame({"a": np.minimum(g_h, g_p), "b": np.maximum(g_h, g_p)})
    e = e[e["a"] != e["b"]].groupby(["a", "b"]).size().rename("c").reset_index()
    e = e[e["c"] >= min_resp]
    G = nx.Graph()
    G.add_nodes_from(set(grupo_de_micro.values()))
    G.add_weighted_edges_from(zip(e["a"].astype(int), e["b"].astype(int), e["c"].astype(float)))
    if G.number_of_edges() == 0:
        return grupo_de_micro, 0
    com = nx.community.louvain_communities(G, weight="weight", resolution=gamma, seed=semilla)
    nuevo = {}
    for c in com:
        rep = min(c)
        for g in c:
            nuevo[g] = rep
    antes = len(set(grupo_de_micro.values()))
    out = {mi: nuevo.get(g, g) for mi, g in grupo_de_micro.items()}
    return out, antes - len(set(out.values()))


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
    cfg = ctx.cfg
    C_all = np.load(ctx.work / "centroides.npy")
    mt = ctx.db.leer("micro_txt")
    com = ctx.db.leer("com")[["comentario_id", "h_texto", "respuesta_a", "n_tok", "norm"]]
    com = com.merge(mt, on="h_texto", how="left")
    corta = int(obtener(cfg, "resp.corta", 3))

    # 1. Ruido estructural por micro-cluster
    st = estadisticas_respuesta(com)
    q_rel = float(obtener(cfg, "ruido.q_rel", 0.15))
    med = st.loc[st.index >= 0, "score_resp"].median()
    st["ruido_micro"] = (st["score_resp"] <= q_rel * med) | (st.index < 0)
    micros_ruido = set(st.index[st["ruido_micro"]])

    # 2. Consolidación de los micro-clusters no ruido
    micro_ids = np.array(sorted(set(st.index) - micros_ruido))
    if len(micro_ids) == 0:  # todo parece ruido: degradar a consolidar todo
        micro_ids, micros_ruido = np.array(sorted(i for i in st.index if i >= 0)), {-1}
    grupo, _, T = consolidar(C_all[micro_ids], float(obtener(cfg, "narr.q_T", 0.6)))
    g_de_m = dict(zip(micro_ids.tolist(), grupo.tolist()))
    n_aglo = len(set(grupo))

    # 3. Fusión por respuestas largas
    ar = aristas_respuesta(com, corta)
    ar = ar[ar["micro"].isin(g_de_m) & ar["micro_p"].isin(g_de_m)]
    g_de_m, uniones = fusionar_por_respuestas(g_de_m, ar, float(obtener(cfg, "narr.gamma_resp", 1.0)),
                                              int(obtener(cfg, "narr.min_resp", 3)))

    # Los micro-clusters ruido van a un grupo propio (sus comentarios respondidos o respuesta se rescatan en e06)
    for mi in micros_ruido:
        g_de_m[mi] = -10_000 - mi
    com["g"] = com["micro"].map(g_de_m)
    orden = com.loc[~com["micro"].isin(micros_ruido), "g"].value_counts().index.tolist()
    orden += [g for g in com["g"].unique() if g not in set(orden)]
    g2lab = {g: f"n{i + 1:03d}" for i, g in enumerate(orden)}
    com["narrativa"] = com["g"].map(g2lab)

    respondidos = set(com.loc[com["respuesta_a"] != "", "respuesta_a"])
    es_resp = com["respuesta_a"] != ""
    max_tok = int(obtener(cfg, "ruido.max_tok", 2))
    com["es_ruido"] = ((com["micro"].isin(micros_ruido) | (com["n_tok"] <= max_tok))
                       & ~es_resp & ~com["comentario_id"].isin(respondidos))

    asig = com[["comentario_id", "micro", "narrativa", "conf", "es_ruido"]].copy()
    micro_narr = st.reset_index()[["micro", "n_com", "ruido_micro", "score_resp"]]
    micro_narr["narrativa"] = micro_narr["micro"].map(com.groupby("micro")["narrativa"].first())

    con_texto = com[~com["es_ruido"]]
    muestra = con_texto.sample(frac=1.0, random_state=0).groupby("narrativa").head(3000)
    docs = muestra.groupby("narrativa")["norm"].apply(" ".join)
    claves = terminos_clave(docs)
    n_por = con_texto["narrativa"].value_counts()
    narr_base = pd.DataFrame([{"narrativa": lab, "n": int(n_por.get(lab, 0)),
                               "titulo": " / ".join(claves.get(lab, [])[:3]) or lab,
                               "frases_clave": "|".join(claves.get(lab, []))}
                              for lab in g2lab.values() if n_por.get(lab, 0) > 0])

    ctx.db.guardar("asig_narr", asig)
    ctx.db.guardar("micro_narr", micro_narr)
    ctx.db.guardar("narr_base", narr_base)
    log.info("e04: %d micro (%d ruido estructural, incl. textos ruido de e02b) -> %d por aglomerativo (T=%.4f) -> %d narrativas "
             "(+%d uniones por respuestas, %d aristas) | ruido=%.1f%%", len(st), len(micros_ruido), n_aglo, T,
             len(narr_base), uniones, len(ar), 100 * com["es_ruido"].mean())
