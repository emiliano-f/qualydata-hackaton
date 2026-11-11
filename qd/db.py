"""Almacenamiento. Interfaz mínima para que las etapas no dependan del motor.

Principal: DuckDB en work/<set>/qd.duckdb (tablas intermedias) + Parquet particionado por día.
Respaldo: AlmacenMemoria (solo pandas) para pruebas de humo cuando DuckDB no está disponible;
no soporta sql(), así que las etapas que necesitan joins pesados exigen DuckDB.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger("qd")


class AlmacenDuckDB:
    motor = "duckdb"

    def __init__(self, ruta: Path, memoria_gb: Optional[float] = None, hilos: Optional[int] = None):
        import duckdb  # import local: el respaldo en memoria funciona sin duckdb instalado

        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(ruta))
        if memoria_gb:
            self.con.execute(f"SET memory_limit='{memoria_gb}GB'")
        if hilos:
            self.con.execute(f"SET threads={int(hilos)}")
        self.con.execute(f"SET temp_directory='{ruta.parent / 'tmp'}'")

    def guardar(self, nombre: str, df: pd.DataFrame) -> None:
        self.con.register("_qd_tmp", df)
        try:
            self.con.execute(f'CREATE OR REPLACE TABLE "{nombre}" AS SELECT * FROM _qd_tmp')
        finally:
            self.con.unregister("_qd_tmp")

    def leer(self, nombre: str) -> pd.DataFrame:
        return self.con.execute(f'SELECT * FROM "{nombre}"').df()

    def existe(self, nombre: str) -> bool:
        q = "SELECT count(*) FROM information_schema.tables WHERE table_name = ?"
        return self.con.execute(q, [nombre]).fetchone()[0] > 0

    def sql(self, consulta: str, params: Optional[list] = None) -> pd.DataFrame:
        return self.con.execute(consulta, params or []).df()

    def ejecutar(self, consulta: str, params: Optional[list] = None) -> None:
        self.con.execute(consulta, params or [])

    def exportar_parquet(self, nombre: str, destino: Path, particion: str) -> None:
        if destino.exists():
            shutil.rmtree(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        self.con.execute(f"COPY \"{nombre}\" TO '{destino}' (FORMAT parquet, PARTITION_BY ({particion}))")

    def cerrar(self) -> None:
        self.con.close()


class AlmacenMemoria:
    motor = "memoria"

    def __init__(self, *_, **__):
        self.tablas: Dict[str, pd.DataFrame] = {}

    def guardar(self, nombre: str, df: pd.DataFrame) -> None:
        self.tablas[nombre] = df.copy()

    def leer(self, nombre: str) -> pd.DataFrame:
        return self.tablas[nombre].copy()

    def existe(self, nombre: str) -> bool:
        return nombre in self.tablas

    def sql(self, consulta: str, params: Optional[list] = None) -> pd.DataFrame:
        raise NotImplementedError("AlmacenMemoria no ejecuta SQL: instalá duckdb")

    def ejecutar(self, consulta: str, params: Optional[list] = None) -> None:
        raise NotImplementedError("AlmacenMemoria no ejecuta SQL: instalá duckdb")

    def exportar_parquet(self, nombre: str, destino: Path, particion: str) -> None:
        log.warning("AlmacenMemoria: se omite la exportación a Parquet de %s", nombre)

    def cerrar(self) -> None:
        self.tablas.clear()


def abrir_almacen(motor: str, work: Path, cfg: dict):
    if motor == "memoria":
        return AlmacenMemoria()
    try:
        return AlmacenDuckDB(work / "qd.duckdb", cfg.get("memoria_gb"), cfg.get("hilos"))
    except ImportError:
        if motor == "duckdb":
            raise
        log.warning("duckdb no está instalado: uso AlmacenMemoria (solo para pruebas)")
        return AlmacenMemoria()
