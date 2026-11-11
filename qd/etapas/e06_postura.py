"""e06 · Postura v0 (sin LLM): léxico + mayoría por micro-cluster + reglas de respuesta.

Orden de la cascada:
  1. Léxico: marcador de DESACUERDO -> rechaza; si no -> apoya.
  2. Mayoría por micro-cluster: si >= postura.micro_mayoria de los comentarios (no respuesta) de un
     micro-cluster tienen marcador de rechazo, el micro-cluster "rechaza" y arrastra a los que no
     tienen marcador de ACUERDO. (1.01 desactiva.)
  3. Respuestas cortas (<= resp.corta tokens), en orden cronológico: heredan la narrativa del padre;
     postura = la del padre, o la opuesta si tienen marcador de desacuerdo.
  4. Ruido: narrativa '' (postura igual se completa, la regla del evaluador lo exige).

Lee asig_ruido (si existe, e05) o asig_narr. Escribe asig(comentario_id, micro, narrativa, conf,
postura, fuente_postura, heredada).
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from qd.config import obtener
from qd.texto import ACUERDO, DESACUERDO

log = logging.getLogger("qd")
SALIDAS = ["asig"]


def patron(marcadores) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(m) for m in sorted(marcadores, key=len, reverse=True)) + r")\b")


RE_DES = patron(DESACUERDO)
RE_ACU = patron(ACUERDO)


def postura_lexica(norm: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"des": norm.str.contains(RE_DES), "acu": norm.str.contains(RE_ACU)}, index=norm.index)


def heredar_respuestas(df: pd.DataFrame, corta: int) -> pd.DataFrame:
    """df con comentario_id, ts, respuesta_a, n_tok, narrativa, postura, des, es_ruido. Orden cronológico."""
    df = df.sort_values("ts", kind="stable")
    narr = dict(zip(df["comentario_id"], df["narrativa"]))
    post = dict(zip(df["comentario_id"], df["postura"]))
    ruido = dict(zip(df["comentario_id"], df["es_ruido"]))
    heredada = {}
    sub = df[(df["respuesta_a"] != "") & (df["n_tok"] <= corta)]
    for cid, padre, des in zip(sub["comentario_id"], sub["respuesta_a"], sub["des"]):
        if padre not in narr or ruido.get(padre, False):
            continue
        narr[cid] = narr[padre]
        p = post[padre]
        post[cid] = ("rechaza" if p == "apoya" else "apoya") if des else p
        ruido[cid] = False
        heredada[cid] = True
    out = df.copy()
    out["narrativa"] = out["comentario_id"].map(narr)
    out["postura"] = out["comentario_id"].map(post)
    out["es_ruido"] = out["comentario_id"].map(ruido)
    out["heredada"] = out["comentario_id"].map(heredada).fillna(False).astype(bool)
    return out


def correr(ctx) -> None:
    base = ctx.db.leer("asig_ruido" if ctx.db.existe("asig_ruido") else "asig_narr")
    if "es_ruido" not in base.columns:
        base["es_ruido"] = base.get("ruido_v0", False)
    com = ctx.db.leer("com")[["comentario_id", "ts", "respuesta_a", "n_tok", "norm"]]
    df = com.merge(base, on="comentario_id", how="left")

    lex = postura_lexica(df["norm"])
    df["des"], df["acu"] = lex["des"], lex["acu"]
    df["postura"] = np.where(df["des"], "rechaza", "apoya")
    df["fuente_postura"] = "lex"

    umbral = float(obtener(ctx.cfg, "postura.micro_mayoria", 0.6))
    no_resp = df[df["respuesta_a"] == ""]
    frac = no_resp.groupby("micro")["des"].mean()
    micros_rech = set(frac[frac >= umbral].index)
    arrastre = df["micro"].isin(micros_rech) & ~df["des"] & ~df["acu"] & (df["respuesta_a"] == "")
    df.loc[arrastre, ["postura", "fuente_postura"]] = ["rechaza", "micro"]

    df = heredar_respuestas(df, int(obtener(ctx.cfg, "resp.corta", 5)))
    df.loc[df["heredada"], "fuente_postura"] = "resp"
    df.loc[df["es_ruido"].astype(bool), "narrativa"] = ""

    asig = df[["comentario_id", "micro", "narrativa", "conf", "postura", "fuente_postura", "heredada"]]
    ctx.db.guardar("asig", asig)
    log.info("e06: rechaza=%.1f%% | micro-clusters 'rechaza'=%d (arrastre %d) | heredadas=%d | ruido=%.1f%%",
             100 * (asig["postura"] == "rechaza").mean(), len(micros_rech), int(arrastre.sum()),
             int(asig["heredada"].sum()), 100 * (asig["narrativa"] == "").mean())
