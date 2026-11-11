"""e09 · Exportación de los 3 CSV de entrega.

Lee lo que exista y completa lo que falte con valores seguros, así `make run` produce
una entrega válida aunque las etapas de narrativas/coordinación todavía no estén:
  asig(comentario_id, narrativa, postura)    -> si falta: todo ruido, postura 'apoya'
  narr(narrativa, titulo, frases_clave, ts_origen)
  grupos(cuenta_id, grupo)                   -> si falta: todas genuinas
Reglas de puntuar.py que se blindan acá: postura nunca vacía, sin 'nan', etiquetas de
comentarios ⊂ narrativas, fecha ISO con -03:00 sin fracción de segundo.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger("qd")

SALIDAS: list = []  # escribe archivos, no tablas: siempre corre
TZ_AR = timezone(timedelta(hours=-3))
POSTURAS = {"apoya", "rechaza"}
BASURA = {"nan", "NaN", "None", "NaT", "<NA>", "null", "NULL"}


def iso_ar(ts) -> str:
    """epoch -> 'YYYY-MM-DDTHH:MM:SS-03:00' (floor al minuto). Lo acepta fromisoformat de Python 3.7+."""
    seg = int(ts) // 60 * 60
    return datetime.fromtimestamp(seg, tz=TZ_AR).strftime("%Y-%m-%dT%H:%M:%S-03:00")


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(object).where(df.notna(), "")
    return df.apply(lambda c: c.map(lambda v: "" if str(v).strip() in BASURA else str(v).strip()))


def armar_comentarios(ids: pd.DataFrame, asig) -> pd.DataFrame:
    out = ids[["comentario_id"]].astype(str)
    if asig is not None and len(asig):
        a = asig[["comentario_id", "narrativa", "postura"]].drop_duplicates("comentario_id")
        out = out.merge(a.astype({"comentario_id": str}), on="comentario_id", how="left")
    else:
        out["narrativa"] = ""
        out["postura"] = ""
    out = limpiar(out)
    out.loc[out["narrativa"].str.lower() == "ninguna", "narrativa"] = ""
    out["postura"] = out["postura"].str.lower()
    out.loc[~out["postura"].isin(POSTURAS), "postura"] = "apoya"  # regla: nunca vacía
    return out


def armar_narrativas(com_out: pd.DataFrame, narr, com: pd.DataFrame) -> pd.DataFrame:
    cols = ["narrativa", "titulo", "frases_clave", "fecha_origen"]
    usadas = set(com_out.loc[com_out["narrativa"] != "", "narrativa"])
    filas = []
    if narr is not None and len(narr):
        n = narr.copy()
        n["narrativa"] = n["narrativa"].astype(str).str.strip()
        n = n[n["narrativa"].isin(usadas)].drop_duplicates("narrativa")
        for r in n.itertuples(index=False):
            ts = getattr(r, "ts_origen", None)
            filas.append({"narrativa": r.narrativa, "titulo": getattr(r, "titulo", "") or r.narrativa,
                          "frases_clave": getattr(r, "frases_clave", "") or "",
                          "fecha_origen": iso_ar(ts) if pd.notna(ts) else ""})
    faltan = usadas - {f["narrativa"] for f in filas}
    if faltan:
        # Etiqueta usada sin fila en narr: origen = primer comentario (respaldo, nunca 0 por ausencia)
        log.warning("%d narrativas sin fila en 'narr': se completan con respaldo", len(faltan))
        t = com_out[com_out["narrativa"].isin(faltan)].merge(com[["comentario_id", "ts"]], on="comentario_id")
        primero = t.groupby("narrativa")["ts"].min()
        for lab in sorted(faltan):
            filas.append({"narrativa": lab, "titulo": lab, "frases_clave": "",
                          "fecha_origen": iso_ar(primero[lab]) if lab in primero else ""})
    out = pd.DataFrame(filas, columns=cols)
    out = limpiar(out) if len(out) else out
    for c in ("titulo", "frases_clave"):
        if len(out):
            out[c] = out[c].str.replace(r"[\r\n]+", " ", regex=True)
    # Sin fecha válida el evaluador da 0 a esa narrativa; mejor el primer comentario que vacío
    vacias = out["fecha_origen"] == "" if len(out) else []
    if len(out) and vacias.any():
        t = com_out.merge(com[["comentario_id", "ts"]], on="comentario_id").groupby("narrativa")["ts"].min()
        out.loc[vacias, "fecha_origen"] = out.loc[vacias, "narrativa"].map(lambda l: iso_ar(t[l]) if l in t else "")
    return out


def armar_cuentas(cta: pd.DataFrame, com: pd.DataFrame, grupos) -> pd.DataFrame:
    todas = pd.Series(pd.concat([cta["cuenta_id"], com["cuenta_id"]]).astype(str).unique(), name="cuenta_id")
    out = todas.to_frame()
    if grupos is not None and len(grupos):
        g = grupos[["cuenta_id", "grupo"]].astype(str).drop_duplicates("cuenta_id")
        out = out.merge(g, on="cuenta_id", how="left")
    else:
        out["grupo"] = ""
    out = limpiar(out)
    return out[out["cuenta_id"] != ""]


def correr(ctx) -> None:
    db = ctx.db
    leer = lambda t: db.leer(t) if db.existe(t) else None  # noqa: E731
    ids = db.leer("ids_entrada")
    com = db.leer("com")
    ctx.costo.comentarios = len(ids)  # también cuando e01 se salteó con --desde

    c = armar_comentarios(ids, leer("asig"))
    n = armar_narrativas(c, leer("narr"), com)
    k = armar_cuentas(db.leer("cta"), com, leer("grupos"))

    ctx.salida.mkdir(parents=True, exist_ok=True)
    opts = dict(index=False, encoding="utf-8", lineterminator="\n")
    c.to_csv(ctx.salida / "comentarios.csv", **opts)
    n.to_csv(ctx.salida / "narrativas.csv", **opts)
    k.to_csv(ctx.salida / "cuentas.csv", **opts)
    log.info("exporta: %d comentarios (%.1f%% ruido), %d narrativas, %d cuentas (%d coordinadas, %d grupos)",
             len(c), 100 * (c["narrativa"] == "").mean(), len(n), len(k),
             int((k["grupo"] != "").sum()), k.loc[k["grupo"] != "", "grupo"].nunique())
