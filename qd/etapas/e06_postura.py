"""e06 · Postura sin LLM: bandos por grafo de respuestas + reglas de respuesta + léxico.

Hechos de la verdad de práctica (no dependen del tema):
  - respuesta corta (<= resp.corta tokens o frase de ACUERDO, con typos) -> misma postura que el padre;
  - respuesta larga -> postura OPUESTA al padre (y siempre misma narrativa).
Entonces, dentro de cada narrativa, las respuestas largas unen micro-clusters de bandos opuestos:
  1. Grafo de micro-clusters con aristas = respuestas largas; partición en 2 bandos por componente
     (vector propio del menor autovalor de la adyacencia normalizada + búsqueda local de corte máximo).
  2. Orientación por votos (postura.voto_*): más NEGACION -> rechaza; más comentarios de cuentas
     coordinadas -> apoya; aparece primero -> apoya.
  3. Micro-clusters sin aristas o mezclados (pureza < postura.pureza_min): léxico DESACUERDO.
  4. Respuestas en orden cronológico: heredan la narrativa del padre; postura por la regla de arriba.
  5. Ruido: narrativa '' (la postura igual se completa: el evaluador la mide sobre la narrativa real).

Escribe asig(comentario_id, micro, narrativa, conf, postura, fuente_postura, heredada) y
bandos_diag (votos de orientación por componente, para calibrar contra la verdad).
"""
from __future__ import annotations

import difflib
import logging
import re

import numpy as np
import pandas as pd

from qd.config import obtener
from qd.texto import ACUERDO, DESACUERDO, NEGACION

log = logging.getLogger("qd")
SALIDAS = ["asig"]


def patron(marcadores) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(m) for m in sorted(marcadores, key=len, reverse=True)) + r")\b")


RE_DES = patron(DESACUERDO)
RE_NEG = patron(NEGACION)
# refutación explícita (más estricta que NEGACION; solo diagnóstico por ahora)
RE_REF = re.compile(r"\b(?:falso|mentira|no es (?:asi|cierto|verdad)|nada que ver|verso|fake|desminti\w*|sin pruebas|"
                    r"no hay pruebas|chamuyo|invent\w+|operacion)\b")
_ACU_POR_LARGO = {}
for _f in ACUERDO:
    _ACU_POR_LARGO.setdefault(len(_f.split()), []).append(_f)


def es_acuerdo(norm: str, corta: int, umbral: float = 0.8) -> bool:
    """Respuesta que acompaña: muy corta, o empieza con una frase de ACUERDO (tolera typos)."""
    w = norm.split()
    if len(w) <= corta:
        return not RE_DES.search(norm)
    if len(w) > 8:  # en práctica, de 9+ tokens casi nunca acompañan
        return False
    for k, frases in _ACU_POR_LARGO.items():
        if len(w) < k:
            continue
        pref = " ".join(w[:k])
        if any((sm := difflib.SequenceMatcher(None, pref, f)).quick_ratio() >= umbral and sm.ratio() >= umbral
               for f in frases):
            return True
    return False


def biparticion(W: np.ndarray, iter_local: int = 20) -> np.ndarray:
    """Max-cut aproximado: signo del vector propio del menor autovalor de D^-1/2 W D^-1/2 + mejora local."""
    n = len(W)
    if n == 1:
        return np.ones(1, dtype=int)
    d = W.sum(1)
    d[d == 0] = 1
    Dm = 1 / np.sqrt(d)
    vals, vecs = np.linalg.eigh(Dm[:, None] * W * Dm[None, :])
    x = np.where(vecs[:, 0] >= 0, 1, -1)
    for _ in range(iter_local):  # mover un nodo si aumenta el peso cortado
        cambio = False
        for i in range(n):
            mismo = W[i, x == x[i]].sum() - W[i, i]
            otro = W[i, x != x[i]].sum()
            if mismo > otro:
                x[i] = -x[i]
                cambio = True
        if not cambio:
            break
    return x


def orientar(b: pd.DataFrame, votos: dict) -> tuple:
    """b: comentarios (no respuesta) de una componente con columnas lado(±1), neg, refuta, coord, ts.
    Devuelve (lado que rechaza, detalle). Votos (puntos > 0 => el lado +1 rechaza):
      refutación explícita ("falso", "mentira", "no es cierto"...): más -> rechaza (poca cobertura, muy precisa)
      negación genérica: más -> rechaza
      cuentas coordinadas: más -> apoya (las campañas empujan la afirmación), solo con diferencia relativa grande
      precedencia: aparece primero -> apoya (peso 0 por defecto; solo desempata)"""
    g = b.groupby("lado")
    neg, coo, t1 = g["neg"].mean(), g["coord"].mean(), g["ts"].quantile(0.01)
    ref = g["refuta"].mean() if "refuta" in b else pd.Series(dtype=float)
    puntos = 0.0
    d_ref = float(ref.get(1, 0) - ref.get(-1, 0)) if len(ref) else 0.0
    if abs(d_ref) >= votos.get("ref_min", 0.02):
        puntos += votos.get("ref", 3.0) * np.sign(d_ref)
    d_neg = neg.get(1, 0) - neg.get(-1, 0)
    if abs(d_neg) > 0.01:
        puntos += votos["neg"] * np.sign(d_neg)
    d_coo = coo.get(1, 0) - coo.get(-1, 0)
    rel = abs(d_coo) / max(float(coo.max()), 1e-9)
    if b["coord"].sum() >= 20 and rel >= votos.get("coord_rel", 0.3):
        puntos -= votos["coord"] * np.sign(d_coo)
    d_t = t1.get(1, 0) - t1.get(-1, 0)
    if abs(d_t) > 3600:
        puntos += votos["temprano"] * np.sign(d_t)
    if puntos == 0 and d_t != 0:  # empate: una narrativa nace como afirmación -> el que aparece primero apoya
        puntos = 0.01 * np.sign(d_t)
    rech = 1 if puntos > 0 else -1
    return rech, {"neg+": round(float(neg.get(1, 0)), 3), "neg-": round(float(neg.get(-1, 0)), 3),
                  "ref+": round(float(ref.get(1, 0)), 3), "ref-": round(float(ref.get(-1, 0)), 3),
                  "coord+": round(float(coo.get(1, 0)), 4), "coord-": round(float(coo.get(-1, 0)), 4),
                  "n+": int((b["lado"] == 1).sum()), "n-": int((b["lado"] == -1).sum()),
                  "dt_h": round(d_t / 3600, 1), "puntos": float(puntos), "rechaza_lado": rech}


def ejemplos_lado(b: pd.DataFrame, lado: int, n: int):
    e = b[b["lado"] == lado].sort_values("conf", ascending=False).drop_duplicates("texto")
    return e["texto"].head(n).tolist()


def bandos_por_narrativa(df: pd.DataFrame, aristas: pd.DataFrame, pureza_min: float, votos: dict,
                         llm=None, n_ej: int = 8, margen: float = 0.1):
    """df: comentarios no ruido con micro, narrativa, ts, neg, coord, es_resp. aristas: micro, micro_p.
    Devuelve ({micro: 'apoya'|'rechaza'}, diagnóstico por componente)."""
    out, diag = {}, []
    narr_de_micro = df.groupby("micro")["narrativa"].agg(lambda s: s.mode().iat[0])
    a = aristas.copy()
    a["narr"] = a["micro"].map(narr_de_micro)
    a = a[a["narr"] == a["micro_p"].map(narr_de_micro)]
    propias = a[a["micro"] == a["micro_p"]].groupby("micro").size()
    cruz = a[a["micro"] != a["micro_p"]]
    base = df[~df["es_resp"]]
    for narr, e in cruz.groupby("narr"):
        nodos = np.array(sorted(set(e["micro"]) | set(e["micro_p"])))
        pos = {m: i for i, m in enumerate(nodos)}
        W = np.zeros((len(nodos), len(nodos)))
        for h, p in zip(e["micro"], e["micro_p"]):
            W[pos[h], pos[p]] += 1
            W[pos[p], pos[h]] += 1
        grado = W.sum(1)
        pureza = grado / (grado + 2 * propias.reindex(nodos).fillna(0).to_numpy())
        # componentes conexas
        comp = -np.ones(len(nodos), dtype=int)
        for s in range(len(nodos)):
            if comp[s] >= 0:
                continue
            pila, comp[s] = [s], s
            while pila:
                i = pila.pop()
                for j in np.flatnonzero(W[i] > 0):
                    if comp[j] < 0:
                        comp[j] = s
                        pila.append(j)
        for c in np.unique(comp):
            idx = np.flatnonzero(comp == c)
            if len(idx) < 2:
                continue
            x = biparticion(W[np.ix_(idx, idx)])
            lado = {nodos[i]: x[k] for k, i in enumerate(idx)}
            b = base[base["micro"].isin(lado)].assign(lado=lambda d: d["micro"].map(lado))
            if b.empty:
                continue
            if b["lado"].nunique() < 2:
                continue
            rech, det = orientar(b, votos)
            det["llm_p+"] = None
            if llm is not None:
                p = llm.orientar(ejemplos_lado(b, 1, n_ej), ejemplos_lado(b, -1, n_ej))
                if p is not None:
                    det["llm_p+"] = round(p, 3)
                    det["llm"] = llm.__class__.__name__
                    det["llm_narrativa"] = getattr(llm, "ultima_narrativa", "")
                    if abs(p - 0.5) >= margen:
                        rech = 1 if p > 0.5 else -1
                        det["rechaza_lado"] = rech
            todos = df[df["micro"].isin(lado)].assign(lado=lambda d: d["micro"].map(lado))
            fr = todos.groupby("lado")["es_resp"].mean()
            det["resp+"], det["resp-"] = round(float(fr.get(1, 0)), 3), round(float(fr.get(-1, 0)), 3)
            diag.append({"narrativa": narr, "componente": int(c), "micros": len(idx), "comentarios": len(b), **det})
            for k, i in enumerate(idx):
                if pureza[i] >= pureza_min:
                    out[nodos[i]] = "rechaza" if x[k] == rech else "apoya"
    return out, pd.DataFrame(diag)


def heredar_respuestas(df: pd.DataFrame, todas: bool = True) -> pd.DataFrame:
    """df con comentario_id, ts, respuesta_a, narrativa, postura, acuerdo, es_ruido. Orden cronológico."""
    df = df.sort_values("ts", kind="stable")
    narr = dict(zip(df["comentario_id"], df["narrativa"]))
    post = dict(zip(df["comentario_id"], df["postura"]))
    ruido = dict(zip(df["comentario_id"], df["es_ruido"]))
    heredada = {}
    sub = df[df["respuesta_a"] != ""]
    if not todas:
        sub = sub[sub["acuerdo"]]
    for cid, padre, acu in zip(sub["comentario_id"], sub["respuesta_a"], sub["acuerdo"]):
        if padre not in narr:
            continue
        narr[cid] = narr[padre]
        p = post[padre]
        post[cid] = p if acu else ("rechaza" if p == "apoya" else "apoya")
        ruido[cid] = False
        ruido[padre] = False  # nadie le responde al ruido
        heredada[cid] = True
    out = df.copy()
    out["narrativa"] = out["comentario_id"].map(narr)
    out["postura"] = out["comentario_id"].map(post)
    out["es_ruido"] = out["comentario_id"].map(ruido)
    out["heredada"] = out["comentario_id"].map(heredada).fillna(False).astype(bool)
    return out


def correr(ctx) -> None:
    cfg = ctx.cfg
    base = ctx.db.leer("asig_ruido" if ctx.db.existe("asig_ruido") else "asig_narr")
    com = ctx.db.leer("com")[["comentario_id", "cuenta_id", "ts", "respuesta_a", "n_tok", "norm", "texto"]]
    df = com.merge(base, on="comentario_id", how="left")
    corta = int(obtener(cfg, "resp.corta", 3))
    df["es_resp"] = df["respuesta_a"] != ""
    df["des"] = df["norm"].str.contains(RE_DES)
    df["neg"] = df["norm"].str.contains(RE_NEG)
    df["refuta"] = df["norm"].str.contains(RE_REF)
    df["acuerdo"] = False
    r = df["es_resp"]
    df.loc[r, "acuerdo"] = [es_acuerdo(t, corta) for t in df.loc[r, "norm"]]

    # 1-3. Bandos por micro-cluster
    coord = set(ctx.db.leer("grupos")["cuenta_id"]) - {""} if ctx.db.existe("grupos") else set()
    df["coord"] = df["cuenta_id"].isin(coord)
    votos = {k: float(obtener(cfg, f"postura.voto_{k}", v)) for k, v in
             (("neg", 1.0), ("coord", 2.0), ("temprano", 0.0))}
    votos["coord_rel"] = float(obtener(cfg, "postura.coord_rel", 0.3))
    votos["ref"] = float(obtener(cfg, "postura.voto_ref", 3.0))
    votos["ref_min"] = float(obtener(cfg, "postura.ref_min", 0.02))
    llm = None
    if not ctx.sin_llm and obtener(cfg, "llm.orientar", True):
        from qd.llm import LLM
        # Aun degradado (sin clave) sirve: chat() responde desde el caché sin red.
        llm = LLM(cfg, ctx.costo, ctx.datos.name)
        ctx.extra["llm"] = llm
    if llm is None and obtener(cfg, "postura.llm_local", False):
        from qd.llm_local import LLMLocal
        llm = LLMLocal(cfg, ctx.costo)
    mic = df[["comentario_id", "micro"]].rename(columns={"comentario_id": "respuesta_a", "micro": "micro_p"})
    aristas = df[r & ~df["acuerdo"]][["micro", "respuesta_a"]].merge(mic, on="respuesta_a")[["micro", "micro_p"]]
    vivos = df[~df["es_ruido"].astype(bool)]
    bando, diag = bandos_por_narrativa(vivos, aristas, float(obtener(cfg, "postura.pureza_min", 0.5)), votos,
                                       llm, int(obtener(cfg, "llm_local.ejemplos", 8)),
                                       float(obtener(cfg, "llm_local.margen", 0.1)))
    if llm is not None and llm.__class__.__name__ == "LLMLocal":
        ctx.costo.registrar_llm(llm.tokens, 0, desde_cache=False, local=True)
        ctx.extra["llm_local"] = llm
    ctx.db.guardar("bandos_diag", diag if len(diag) else pd.DataFrame({"narrativa": [""]}))
    df["postura"] = np.where(df["des"], "rechaza", "apoya")
    df["fuente_postura"] = "lex"
    con_bando = df["micro"].isin(bando) & ~df["es_resp"]
    df.loc[con_bando, "postura"] = df.loc[con_bando, "micro"].map(bando)
    df.loc[con_bando, "fuente_postura"] = "bando"

    # 4. Respuestas
    df = heredar_respuestas(df, bool(obtener(cfg, "resp.heredar_todas", True)))
    df.loc[df["heredada"], "fuente_postura"] = "resp"
    df.loc[df["es_ruido"].astype(bool), "narrativa"] = ""

    asig = df[["comentario_id", "micro", "narrativa", "conf", "postura", "fuente_postura", "heredada"]]
    ctx.db.guardar("asig", asig)
    fp = asig["fuente_postura"].value_counts(normalize=True).mul(100).round(1).to_dict()
    log.info("e06: rechaza=%.1f%% | micro-clusters con bando=%d | fuentes %s | heredadas=%d | ruido=%.1f%%",
             100 * (asig["postura"] == "rechaza").mean(), len(bando), fp, int(asig["heredada"].sum()),
             100 * (asig["narrativa"] == "").mean())
