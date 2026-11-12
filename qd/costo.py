"""Registro automático de tiempo, tokens y USD para costo.json."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

try:
    import resource  # solo Unix; en Windows ram_pico_gb queda en null
except ImportError:  # pragma: no cover
    resource = None


def _ram_gb() -> Optional[int]:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 2**30)
    except ImportError:
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
        except (ValueError, OSError, AttributeError):
            return None


def _gpu() -> Optional[str]:
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - torch roto no debe tirar el pipeline
        return None


def describir_hardware() -> str:
    cpu = platform.processor() or platform.machine() or "CPU"
    ram = _ram_gb()
    gpu = _gpu()
    partes = [f"{platform.system()} {cpu}", f"{os.cpu_count()} núcleos"]
    if ram:
        partes.append(f"{ram} GB RAM")
    partes.append(f"GPU {gpu}" if gpu else "sin GPU")
    return ", ".join(partes)


def _ram_pico_gb():
    """Pico de RSS del proceso: ru_maxrss viene en KB en Linux y en bytes en macOS."""
    if resource is None:
        return None
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(pico / 1e9, 2) if sys.platform == "darwin" else round(pico / 1e6, 2)


def medir_almacenamiento(datos: Path, work: Path) -> dict:
    """Tamaños reales en MB (1e6 bytes): la base DuckDB (+ WAL si existe), el parquet
    particionado por día y la entrada completa menos la carpeta verdad/."""
    def _tam(ruta: Path) -> float:
        if ruta.is_file():
            return ruta.stat().st_size / 1e6
        if ruta.is_dir():
            return sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file()) / 1e6
        return 0.0

    db = work / "qd.duckdb"
    entrada = sum(f.stat().st_size for f in datos.rglob("*")
                  if f.is_file() and "verdad" not in f.relative_to(datos).parts) / 1e6
    return {"duckdb": round(_tam(db) + _tam(Path(str(db) + ".wal")), 1),
            "parquet": round(_tam(work / "parquet"), 1),
            "entrada": round(entrada, 1)}


class Costo:
    def __init__(self, precios: Dict[str, float], almacenamiento: str):
        self.t0 = time.perf_counter()
        self.etapas: Dict[str, float] = {}
        self.precios = precios
        self.almacenamiento = almacenamiento
        self.tok_in = 0
        self.tok_out = 0
        self.tok_local = 0
        self.llamadas = 0            # llamadas de red al LLM
        self.aciertos_cache = 0
        self.fallas_llm = 0          # llamadas de red fallidas (incl. reintentos agotados)
        self.faltas_cache = 0        # respuestas pedidas que no estaban en caché (sin clave / degradado)
        self.llm_usado = False       # se instanció algún LLM por API
        self.almacenamiento_mb = None  # lo mide el pipeline justo antes de escribir costo.json
        self.modo_llm = "apagado"
        self.notas = []
        self.comentarios = 0

    @contextmanager
    def etapa(self, nombre: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.etapas[nombre] = round(time.perf_counter() - t, 2)

    def registrar_llm(self, tok_in: int, tok_out: int, desde_cache: bool, local: bool = False) -> None:
        """tokens_ia = lo que costó producir las respuestas usadas: en acierto de caché, el uso
        guardado en el archivo; en llamada real, el uso real. Los de un LLM local cuentan pero cuestan 0."""
        if local:
            self.tok_local += int(tok_in or 0) + int(tok_out or 0)
            return
        if desde_cache:
            self.aciertos_cache += 1
        else:
            self.llamadas += 1
        self.tok_in += int(tok_in or 0)
        self.tok_out += int(tok_out or 0)

    def resolver_modo_llm(self) -> None:
        """modo_llm ∈ {apagado, degradado, desde_cache, completo}: apagado = no se instanció LLM;
        degradado = alguna llamada de red falló o alguna respuesta pedida no estaba en caché;
        desde_cache = todas vinieron del caché (0 llamadas, >= 1 acierto); completo = resto."""
        if not self.llm_usado:
            self.modo_llm = "apagado"
        elif self.fallas_llm or self.faltas_cache:
            self.modo_llm = "degradado"
            if self.faltas_cache:
                self.nota(f"sin LLM_API_KEY: {self.faltas_cache} respuestas no estaban en caché, "
                          "se usó el respaldo determinista")
        elif self.llamadas == 0 and self.aciertos_cache > 0:
            self.modo_llm = "desde_cache"
        else:
            self.modo_llm = "completo"

    def nota(self, texto: str) -> None:
        if texto not in self.notas:
            self.notas.append(texto)

    def usd(self) -> float:
        return (self.tok_in / 1e6 * float(self.precios.get("usd_por_millon_entrada", 0))
                + self.tok_out / 1e6 * float(self.precios.get("usd_por_millon_salida", 0)))

    def a_dict(self) -> dict:
        self.resolver_modo_llm()
        if self.modo_llm == "apagado":
            self.nota("sin IA en esta corrida")
        usd = self.usd()
        seg = round(time.perf_counter() - self.t0, 1)
        n = max(self.comentarios, 1)
        usd_hora = self.precios.get("usd_hora_cpu")
        return {
            "comentarios_procesados": self.comentarios,
            "segundos": seg,
            "tokens_ia": self.tok_in + self.tok_out + self.tok_local,
            "usd_ia": round(usd, 4),
            "usd_por_mil_comentarios": round(usd / n * 1000, 6),
            "hardware": describir_hardware(),
            "almacenamiento": self.almacenamiento,
            "notas": "; ".join(self.notas),
            # Extras (el evaluador los ignora; sirven al jurado)
            "ram_pico_gb": _ram_pico_gb(),
            "almacenamiento_mb": self.almacenamiento_mb,
            "comentarios_por_segundo": round(self.comentarios / seg, 1) if seg else None,
            "segundos_por_mil_comentarios": round(seg / n * 1000, 4),
            "usd_computo_por_mil": round(seg * float(usd_hora) / 3600 / n * 1000, 6) if usd_hora else None,
            "proyeccion_1M_por_dia": {
                "horas": round(seg / n * 1e6 / 3600, 2),
                "metodo": "extrapolación lineal desde esta corrida; estimación, no medición: "
                          "e08 crece más que linealmente en publicaciones virales"},
            "desglose_segundos": self.etapas,
            "llamadas_llm": self.llamadas,
            "tokens_llm_local": self.tok_local,
            "aciertos_cache_llm": self.aciertos_cache,
            "modo_llm": self.modo_llm,
            "modelo_llm": self.precios.get("modelo", ""),
        }

    def escribir(self, ruta: Path) -> dict:
        datos = self.a_dict()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
        return datos
