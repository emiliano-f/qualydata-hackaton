"""Carga de config.yaml y precios.yaml."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def cargar(ruta: Path) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def obtener(cfg: dict, ruta: str, defecto: Any = None) -> Any:
    """obtener(cfg, 'narr.q_T', 0.6)"""
    nodo: Any = cfg
    for parte in ruta.split("."):
        if not isinstance(nodo, dict) or parte not in nodo:
            return defecto
        nodo = nodo[parte]
    return nodo


def huella(cfg: dict) -> str:
    """Hash corto de la configuración, para experimentos.csv."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
