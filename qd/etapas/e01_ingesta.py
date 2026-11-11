"""e01 · Ingesta: CSV crudos -> tablas limpias en el almacén (+ Parquet particionado por día).

Tablas que escribe (contrato):
  ids_entrada(comentario_id)                         universo a cubrir en comentarios.csv
  com(comentario_id, publicacion_id, cuenta_id, ts, dia, texto, respuesta_a, me_gusta,
      norm, n_tok, h_texto, h_esq)                   1 fila por comentario válido
                                                     (respuesta_a = '' si no hay padre válido)
  com_descartados(comentario_id, motivo)
  textos(h_texto, norm, n, dia_min)                  1 fila por texto normalizado único
  pub(publicacion_id, fuente_id, ts, titular)
  cta(cuenta_id, usuario, ts_creacion, seguidores, siguiendo, edad_dias)
  fue(fuente_id, nombre, tipo)
Tiempos: epoch en segundos (BIGINT). La zona -03:00 se aplica solo al exportar.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from qd import entrada
from qd.texto import esqueleto, hash64, normalizar

log = logging.getLogger("qd")

SALIDAS = ["ids_entrada", "com", "com_descartados", "textos", "pub", "cta", "fue"]
OFFSET_AR = -3 * 3600


def _numero(serie: pd.Series, defecto: int = 0) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce").fillna(defecto).astype("int64")


def construir_com(raw: pd.DataFrame):
    """Devuelve (ids_entrada, com, descartados). Pura pandas: testeable sin almacén."""
    raw = raw[raw["comentario_id"] != ""]
    ids = pd.DataFrame({"comentario_id": raw["comentario_id"].unique()})

    dup = raw["comentario_id"].duplicated(keep="first")
    desc = [pd.DataFrame({"comentario_id": raw.loc[dup, "comentario_id"], "motivo": "id_duplicado"})]
    df = raw.loc[~dup].copy()

    df["ts"] = entrada.a_epoch(df["fecha"])
    malas = df["ts"].isna()
    desc.append(pd.DataFrame({"comentario_id": df.loc[malas, "comentario_id"], "motivo": "fecha_invalida"}))
    df = df.loc[~malas].copy()
    df["ts"] = df["ts"].astype("int64")

    df["me_gusta"] = _numero(df["me_gusta"]).clip(lower=0)
    df["texto"] = df["texto"].astype(str)

    validos = set(df["comentario_id"])
    resp = df["respuesta_a"].astype(str).str.strip()
    huerfanas = (resp != "") & ~resp.isin(validos)
    if huerfanas.any():
        log.info("respuesta_a apunta a %d comentarios inexistentes: se anulan", int(huerfanas.sum()))
    # '' en vez de NULL: evita que DuckDB infiera tipo NULL si la columna viene entera vacía
    df["respuesta_a"] = resp.where((resp != "") & ~huerfanas, "")

    dia_local = (df["ts"] + OFFSET_AR) // 86400
    df["dia"] = (dia_local - dia_local.min()).astype("int64") if len(df) else dia_local

    # Normalización sobre textos únicos (el set sintético repite mucho)
    unicos = pd.Series(df["texto"].unique())
    norm = unicos.map(normalizar)
    mapa = pd.DataFrame({"texto": unicos, "norm": norm})
    mapa["n_tok"] = mapa["norm"].str.split().str.len().fillna(0).astype("int64")
    mapa["h_texto"] = mapa["norm"].map(hash64).astype("int64")
    mapa["h_esq"] = mapa["norm"].map(lambda s: hash64(esqueleto(s))).astype("int64")
    df = df.merge(mapa, on="texto", how="left")

    columnas = ["comentario_id", "publicacion_id", "cuenta_id", "ts", "dia", "texto", "respuesta_a",
                "me_gusta", "norm", "n_tok", "h_texto", "h_esq"]
    com = df[columnas].reset_index(drop=True)
    descartados = pd.concat(desc, ignore_index=True)
    return ids, com, descartados


def construir_textos(com: pd.DataFrame) -> pd.DataFrame:
    g = com.groupby("h_texto", sort=False)
    return pd.DataFrame({"norm": g["norm"].first(), "n": g.size(), "dia_min": g["dia"].min()}).reset_index()


def construir_cta(raw_cta: pd.DataFrame, com: pd.DataFrame) -> pd.DataFrame:
    cta = raw_cta[raw_cta["cuenta_id"] != ""].drop_duplicates("cuenta_id").copy()
    # Cuentas que comentan pero no están en cuentas.csv: se agregan sin metadatos
    extra = sorted(set(com["cuenta_id"]) - set(cta["cuenta_id"]))
    if extra:
        log.info("%d cuentas comentan pero no figuran en cuentas.csv", len(extra))
        cta = pd.concat([cta, pd.DataFrame({"cuenta_id": extra})], ignore_index=True)
    out = pd.DataFrame({"cuenta_id": cta["cuenta_id"].astype(str),
                        "usuario": cta["usuario"].fillna("").astype(str)})
    out["ts_creacion"] = entrada.a_epoch(cta["fecha_creacion"].fillna("")).to_numpy()
    out["seguidores"] = _numero(cta["seguidores"]).to_numpy()
    out["siguiendo"] = _numero(cta["siguiendo"]).to_numpy()
    t0 = int(com["ts"].min()) if len(com) else 0
    out["edad_dias"] = ((t0 - out["ts_creacion"].astype("float64")) / 86400).to_numpy()
    return out


def correr(ctx) -> None:
    raw = entrada.leer(ctx.datos, "comentarios")
    ids, com, descartados = construir_com(raw)
    if com.empty:
        raise ValueError("No quedó ningún comentario válido tras la ingesta")

    pub = entrada.leer(ctx.datos, "publicaciones", obligatorio=False)
    pub = pub[pub["publicacion_id"] != ""].drop_duplicates("publicacion_id")
    pub = pd.DataFrame({"publicacion_id": pub["publicacion_id"], "fuente_id": pub["fuente_id"],
                        "ts": entrada.a_epoch(pub["fecha"]), "titular": pub["titular"]})
    fue = entrada.leer(ctx.datos, "fuentes", obligatorio=False).drop_duplicates("fuente_id")
    cta = construir_cta(entrada.leer(ctx.datos, "cuentas", obligatorio=False), com)
    textos = construir_textos(com)

    for nombre, df in [("ids_entrada", ids), ("com", com), ("com_descartados", descartados),
                       ("textos", textos), ("pub", pub), ("cta", cta), ("fue", fue)]:
        ctx.db.guardar(nombre, df)

    if ctx.cfg.get("almacen", {}).get("parquet", True):
        try:
            ctx.db.exportar_parquet("com", ctx.work / "parquet" / "comentarios", "dia")
        except Exception as e:  # noqa: BLE001 - el Parquet es para el relato de escala, no es crítico
            log.warning("No se pudo exportar Parquet: %s", e)

    ctx.costo.comentarios = len(ids)
    log.info("ingesta: %d comentarios (%d válidos, %d descartados), %d textos únicos (%.0f%%), %d cuentas",
             len(ids), len(com), len(descartados), len(textos), 100 * len(textos) / len(com), len(cta))
