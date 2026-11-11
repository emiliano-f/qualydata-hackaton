"""Orquestador: `python -m qd.pipeline --datos datos/practica --salida salida/practica`.

Cada etapa es un módulo con SALIDAS (tablas que escribe) y correr(ctx).
--desde eNN saltea las etapas anteriores si sus tablas ya existen (loop en caliente).
Para agregar una etapa: crear qd/etapas/eNN_nombre.py e insertarla en ETAPAS.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from qd import config as qcfg
from qd.costo import Costo
from qd.db import abrir_almacen

ETAPAS = [
    "e01_ingesta",
    "e02_embed",
    "e03_micro",
    "e04_narrativas",
    # "e05_ruido",
    "e06_postura",
    "e07_origen",
    # "e08_coord",
    "e09_exporta",
]

log = logging.getLogger("qd")


@dataclass
class Contexto:
    datos: Path
    salida: Path
    work: Path
    cfg: dict
    db: object
    costo: Costo
    sin_llm: bool = False
    embed: str = "e5"
    extra: dict = field(default_factory=dict)


def configurar_log(salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(salida / "pipeline.log", mode="w",
                                                                     encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--precios", type=Path, default=Path("precios.yaml"))
    ap.add_argument("--work", type=Path, default=None, help="por defecto work/<nombre de --datos>")
    ap.add_argument("--desde", default=None, help="ej. e04: recalcula desde esa etapa")
    ap.add_argument("--almacen", choices=["auto", "duckdb", "memoria"], default="auto")
    ap.add_argument("--sin-llm", action="store_true")
    ap.add_argument("--embed", choices=["e5", "tfidf"], default=None)
    ap.add_argument("--no-validar", action="store_true")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="CLAVE=VALOR",
                    help="pisa config.yaml, ej. --set narr.q_T=0.5 (repetible)")
    a = ap.parse_args(argv)

    configurar_log(a.salida)
    cfg = qcfg.aplicar_sets(qcfg.cargar(a.config), a.sets)
    precios = qcfg.cargar(a.precios) if a.precios.exists() else {}
    work = a.work or Path("work") / a.datos.name
    work.mkdir(parents=True, exist_ok=True)

    db = abrir_almacen(a.almacen, work, cfg.get("almacen", {}))
    desc = "DuckDB local + Parquet particionado por día" if db.motor == "duckdb" else "pandas en memoria (prueba)"
    costo = Costo(precios, desc)
    sin_llm = a.sin_llm or not os.environ.get("LLM_API_KEY") or not qcfg.obtener(cfg, "llm.activo", True)
    costo.modo_llm = "apagado" if sin_llm else "completo"
    ctx = Contexto(a.datos, a.salida, work, cfg, db, costo, sin_llm=sin_llm,
                   embed=a.embed or qcfg.obtener(cfg, "embed.motor", "e5"))
    log.info("pipeline %s | datos=%s salida=%s almacén=%s llm=%s config=%s", "v" + __import__("qd").__version__,
             a.datos, a.salida, db.motor, costo.modo_llm, qcfg.huella(cfg))

    if a.desde and not any(e.startswith(a.desde) for e in ETAPAS):
        log.error("--desde %s no coincide con ninguna etapa %s", a.desde, ETAPAS)
        return 2
    saltear = a.desde is not None
    try:
        for nombre in ETAPAS:
            mod = importlib.import_module(f"qd.etapas.{nombre}")
            if saltear and nombre.startswith(a.desde):
                saltear = False
            salidas = getattr(mod, "SALIDAS", [])
            if saltear and salidas and all(db.existe(t) for t in salidas):
                log.info("· %s (salteada: tablas en caché)", nombre)
                continue
            t = time.perf_counter()
            with costo.etapa(nombre):
                mod.correr(ctx)
            log.info("✓ %s en %.1fs", nombre, time.perf_counter() - t)
    except Exception:
        log.exception("El pipeline falló")
        return 1
    finally:
        if costo.modo_llm == "apagado":
            costo.nota("sin IA en esta corrida")
        costo.escribir(a.salida / "costo.json")
        db.cerrar()

    if a.no_validar:
        return 0
    from qd.validar import validar

    err, av = validar(a.datos, a.salida, cfg.get("sanidad", {}) or {})
    for x in av:
        log.warning("validación: %s", x)
    for x in err:
        log.error("validación: %s", x)
    log.info("listo en %.1fs → %s", time.perf_counter() - costo.t0, a.salida)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
