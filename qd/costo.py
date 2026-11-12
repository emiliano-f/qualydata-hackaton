"""Registro automático de tiempo, tokens y USD para costo.json."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional


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
        n = max(self.comentarios, 1)
        return {
            "comentarios_procesados": self.comentarios,
            "segundos": round(time.perf_counter() - self.t0, 1),
            "tokens_ia": self.tok_in + self.tok_out + self.tok_local,
            "usd_ia": round(usd, 4),
            "usd_por_mil_comentarios": round(usd / n * 1000, 6),
            "hardware": describir_hardware(),
            "almacenamiento": self.almacenamiento,
            "notas": "; ".join(self.notas),
            # Extras (el evaluador los ignora; sirven al jurado)
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
