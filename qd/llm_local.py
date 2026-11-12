"""LLM local chico (CPU, sin clave, USD 0) para decisiones puntuales.

- orientar(lado_a, lado_b): ¿qué bando RECHAZA la afirmación que circula? Lee los logits del primer
  token ("A" / "B") en las dos posiciones (para anular el sesgo de orden) y devuelve P(lado_a rechaza).
- titular(ejemplos): título corto de una narrativa (generación greedy de ~24 tokens).
Todas las respuestas se cachean en disco (reproducible y gratis en corridas repetidas).
Cualquier error devuelve None y el pipeline sigue con su respaldo determinista.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from qd.config import obtener

log = logging.getLogger("qd")

SISTEMA = "Sos un analista de discurso en redes sociales en español rioplatense. Respondés con precisión y sin explicar."
P_ORIENTAR = """Estos comentarios discuten una misma narrativa (una afirmación que circula). Los grupos A y B tienen posturas OPUESTAS: uno SOSTIENE la afirmación (la instala, la repite, la amplifica, la denuncia) y el otro la RECHAZA (la desmiente, la contradice o defiende lo contrario).

Grupo A:
{a}

Grupo B:
{b}

¿Qué grupo RECHAZA la afirmación que circula? Respondé solo con una letra: A o B."""
P_TITULO = """Estos comentarios sostienen una misma afirmación:
{a}

Escribí la afirmación en una sola frase corta (máximo 12 palabras), sin comillas."""


class LLMLocal:
    def __init__(self, cfg: dict, costo=None):
        self.cfg = cfg
        self.costo = costo
        self.nombre = obtener(cfg, "llm_local.modelo", "Qwen/Qwen2.5-1.5B-Instruct")
        self.cache = Path(obtener(cfg, "llm_local.cache_dir", "cache/llm_local"))
        self.cache.mkdir(parents=True, exist_ok=True)
        self._mod = self._tok = None
        self.roto = False
        self.tokens = 0

    def _cargar(self) -> bool:
        if self._mod is not None or self.roto:
            return not self.roto
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            t = time.perf_counter()
            self._tok = AutoTokenizer.from_pretrained(self.nombre)
            try:
                self._mod = AutoModelForCausalLM.from_pretrained(self.nombre, dtype=torch.float32)
            except TypeError:  # transformers viejos
                self._mod = AutoModelForCausalLM.from_pretrained(self.nombre, torch_dtype=torch.float32)
            self._mod.eval()
            self._torch = torch
            log.info("llm_local: %s cargado en %.0fs", self.nombre, time.perf_counter() - t)
            if self.costo:
                self.costo.nota(f"LLM local {self.nombre} en CPU (USD 0) para orientar bandos")
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("llm_local no disponible (%s): se usan los votos heurísticos", e)
            self.roto = True
            return False

    def _clave(self, *partes) -> Path:
        h = hashlib.sha256(json.dumps([self.nombre, *partes], ensure_ascii=False).encode()).hexdigest()[:24]
        return self.cache / f"{h}.json"

    def _entrada(self, prompt: str):
        msgs = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": prompt}]
        texto = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self._tok(texto, return_tensors="pt")

    def _p_a(self, prompt: str) -> float:
        ids = self._entrada(prompt)
        self.tokens += int(ids["input_ids"].shape[1]) + 1
        with self._torch.no_grad():
            logits = self._mod(**ids).logits[0, -1]
        ta = self._tok.encode("A", add_special_tokens=False)[0]
        tb = self._tok.encode("B", add_special_tokens=False)[0]
        par = self._torch.softmax(self._torch.stack([logits[ta], logits[tb]]), dim=0)
        return float(par[0])

    @staticmethod
    def _lista(ejemplos: List[str]) -> str:
        return "\n".join(f"- {e.strip()[:180]}" for e in ejemplos)

    def orientar(self, lado_a: List[str], lado_b: List[str]) -> Optional[float]:
        """P(lado_a es el que rechaza). Promedia las dos posiciones."""
        f = self._clave("orientar", lado_a, lado_b)
        if f.exists():
            return json.loads(f.read_text())["p"]
        if not self._cargar():
            return None
        try:
            p1 = self._p_a(P_ORIENTAR.format(a=self._lista(lado_a), b=self._lista(lado_b)))       # A = lado_a
            p2 = 1 - self._p_a(P_ORIENTAR.format(a=self._lista(lado_b), b=self._lista(lado_a)))   # B = lado_a
            p = (p1 + p2) / 2
            f.write_text(json.dumps({"p": p, "p1": p1, "p2": p2}))
            return p
        except Exception as e:  # noqa: BLE001
            log.warning("llm_local.orientar falló: %s", e)
            return None

    def titular(self, ejemplos: List[str]) -> Optional[str]:
        f = self._clave("titular", ejemplos)
        if f.exists():
            return json.loads(f.read_text())["t"]
        if not self._cargar():
            return None
        try:
            ids = self._entrada(P_TITULO.format(a=self._lista(ejemplos)))
            with self._torch.no_grad():
                out = self._mod.generate(**ids, max_new_tokens=24, do_sample=False)
            nuevo = out[0, ids["input_ids"].shape[1]:]
            self.tokens += int(ids["input_ids"].shape[1]) + len(nuevo)
            t = self._tok.decode(nuevo, skip_special_tokens=True).strip().strip('"').splitlines()[0][:140]
            f.write_text(json.dumps({"t": t}, ensure_ascii=False))
            return t or None
        except Exception as e:  # noqa: BLE001
            log.warning("llm_local.titular falló: %s", e)
            return None
