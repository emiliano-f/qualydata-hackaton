"""Cliente de LLM por API (endpoint compatible con OpenAI: /chat/completions). Solo biblioteca estándar.

Sirve para OpenAI, Gemini (generativelanguage.googleapis.com/v1beta/openai/), Anthropic
(api.anthropic.com/v1/) y otros compatibles: se configura llm.base_url y llm.modelo; la clave va en la
variable de entorno LLM_API_KEY (nunca en el repo).

- Caché en disco por hash(modelo + prompt): corridas repetidas no pagan y el jurado reproduce sin clave
  si se commitea cache/llm/<set>/.
- Reintentos con backoff ante 429/5xx/timeout; corte (modo degradado) tras llm.max_fallas fallas
  seguidas o al superar llm.tope_tokens. Cualquier falla devuelve None y el pipeline sigue con su respaldo.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from qd.config import obtener

log = logging.getLogger("qd")

SISTEMA = ("Sos un analista de discurso en redes sociales argentinas. Respondés exactamente en el formato "
           "pedido, sin explicaciones.")
P_ORIENTAR = """Estos comentarios discuten una misma narrativa (una afirmación o encuadre que circula). Los grupos A y B tienen posturas OPUESTAS: uno SOSTIENE la afirmación que circula (la instala, la repite, la amplifica, la denuncia) y el otro la RECHAZA (la desmiente, la contradice o defiende lo contrario).

Grupo A:
{a}

Grupo B:
{b}

¿Qué grupo RECHAZA la afirmación que circula? Respondé solo con una letra: A o B."""
P_ORIENTAR_V2 = """{ejemplos}Ahora una narrativa NUEVA. Los grupos A y B la discuten con posturas OPUESTAS: uno la SOSTIENE y el otro la RECHAZA.

Grupo A:
{a}

Grupo B:
{b}

Primero escribí la narrativa en una línea, como afirmación, en el mismo estilo que los ejemplos. Después indicá qué grupo la RECHAZA.
Respondé exactamente con este formato:
NARRATIVA: <una línea>
RECHAZA: <A o B>"""
P_TITULO = """Estos comentarios sostienen una misma afirmación que circula en redes:
{a}

Devolvé un JSON con dos campos: "titulo" (la afirmación en una frase corta, máximo 12 palabras, sin comillas internas) y "frases_clave" (lista de 4 a 6 términos o frases cortas que la identifican)."""


class LLM:
    def __init__(self, cfg: dict, costo=None, set_nombre: str = "set"):
        self.cfg = cfg
        self.costo = costo
        if costo is not None:
            costo.llm_usado = True
        self.clave = os.environ.get("LLM_API_KEY", "")
        self.base = str(obtener(cfg, "llm.base_url", "https://api.openai.com/v1/")).rstrip("/") + "/"
        self.modelo = str(obtener(cfg, "llm.modelo", ""))
        self.timeout = float(obtener(cfg, "llm.timeout_s", 60))
        self.reintentos = int(obtener(cfg, "llm.reintentos", 3))
        self.max_fallas = int(obtener(cfg, "llm.max_fallas", 4))
        self.tope = int(obtener(cfg, "llm.tope_tokens", 2_000_000))
        self.reasoning = str(obtener(cfg, "llm.reasoning_effort", "") or "")
        self.cache = Path(obtener(cfg, "llm.cache_dir", "cache/llm")) / set_nombre
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fallas_seguidas = 0
        self.degradado = not (self.clave and self.modelo)
        if self.degradado:
            log.info("LLM: %s -> solo caché (%s)",
                     "sin LLM_API_KEY" if not self.clave else "llm.modelo vacío", self.cache)
        self.tokens = 0

    def disponible(self) -> bool:
        return not self.degradado

    def _post(self, cuerpo: dict) -> dict:
        req = urllib.request.Request(self.base + "chat/completions", data=json.dumps(cuerpo).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.clave}"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    @staticmethod
    def _uso_tokens(uso: dict) -> tuple:
        """(tokens de entrada, de salida) de un usage de la API. Salida = total - prompt si viene
        total_tokens (incluye razonamiento oculto, que se factura); si no, completion_tokens."""
        ti = int(uso.get("prompt_tokens", 0) or 0)
        if "total_tokens" in uso:
            return ti, int(uso.get("total_tokens") or 0) - ti
        return ti, int(uso.get("completion_tokens", 0) or 0)

    def chat(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        f = self.cache / (hashlib.sha256(json.dumps([self.modelo, SISTEMA, prompt, max_tokens, self.reasoning],
                                                    ensure_ascii=False).encode()).hexdigest()[:32] + ".json")
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            if self.costo:
                self.costo.registrar_llm(*self._uso_tokens(d.get("uso") or {}), desde_cache=True)
            return d["texto"]
        if self.degradado or self.tokens >= self.tope:
            if self.costo:
                self.costo.faltas_cache += 1  # respuesta pedida que no estaba en caché
            return None
        cuerpo = {"model": self.modelo, "temperature": 0, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": prompt}]}
        if self.reasoning:
            cuerpo["reasoning_effort"] = self.reasoning
        espera = 2.0
        for intento in range(self.reintentos + 1):
            try:
                r = self._post(cuerpo)
                ch = r["choices"][0]
                texto = (ch["message"].get("content") or "").strip()
                uso = r.get("usage") or {}
                ti, to = self._uso_tokens(uso)
                self.tokens += ti + to
                if self.costo:
                    self.costo.registrar_llm(ti, to, desde_cache=False)
                if ch.get("finish_reason") == "length" or not texto:
                    # Razonamiento oculto comió max_tokens (finish_reason=length) o vino vacía:
                    # no se usa, no se cachea y no se reintenta el mismo prompt; cuenta como falla
                    # pero los tokens igual se facturaron (ya registrados arriba).
                    log.warning("LLM: respuesta truncada/vacía (finish_reason=%s); no se cachea ni se reintenta",
                                ch.get("finish_reason"))
                    if self.costo:
                        self.costo.fallas_llm += 1
                    break
                f.write_text(json.dumps({"texto": texto, "uso": uso}, ensure_ascii=False), encoding="utf-8")
                self.fallas_seguidas = 0
                return texto
            except urllib.error.HTTPError as e:
                if self.costo:
                    self.costo.fallas_llm += 1
                reintentable = e.code == 429 or e.code >= 500
                log.warning("LLM HTTP %s (intento %d): %s", e.code, intento + 1, e.read()[:200])
                if not reintentable:
                    break
            except Exception as e:  # noqa: BLE001 - timeout, red, respuesta rara
                if self.costo:
                    self.costo.fallas_llm += 1
                log.warning("LLM error (intento %d): %s", intento + 1, e)
            time.sleep(espera)
            espera *= 2
        self.fallas_seguidas += 1
        if self.fallas_seguidas >= self.max_fallas:
            log.warning("LLM: %d fallas seguidas -> modo degradado (respaldo determinista)", self.fallas_seguidas)
            self.degradado = True
        return None

    @staticmethod
    def _lista(ejemplos: List[str]) -> str:
        return "\n".join(f"- {e.strip()[:220]}" for e in ejemplos)

    @staticmethod
    def _letra(texto: Optional[str]) -> Optional[str]:
        if not texto:
            return None
        m = re.search(r"\b([AB])\b", texto.strip().upper())
        return m.group(1) if m else None

    def _bloque_ejemplos(self) -> str:
        """Ejemplos resueltos de la verdad de práctica (qd/fewshot_postura.json, ver herramientas/fewshot.py)."""
        if not hasattr(self, "_ej"):
            f = Path(__file__).with_name("fewshot_postura.json")
            ej = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
            partes = []
            for i, e in enumerate(ej, 1):
                partes.append(f"Ejemplo {i}. Narrativa: «{e['narrativa']}»\nComentarios que la SOSTIENEN:\n"
                              f"{self._lista(e['apoyan'])}\nComentarios que la RECHAZAN:\n{self._lista(e['rechazan'])}\n")
            self._ej = ("Así se distingue la narrativa (lo que se SOSTIENE) de su rechazo, en ejemplos resueltos:\n\n"
                        + "\n".join(partes) + "\n") if partes else ""
        return self._ej

    @staticmethod
    def _rechaza(texto: Optional[str]):
        if not texto:
            return None, ""
        m = re.search(r"RECHAZA\s*:\s*\**\s*([AB])\b", texto, re.I)
        letra = m.group(1).upper() if m else LLM._letra(texto)
        if letra is None:
            log.warning("orientar: no se encontró la letra RECHAZA en %r", texto[:160])
        n = re.search(r"NARRATIVA\s*:\s*(.+)", texto, re.I)
        return letra, (n.group(1).strip().strip("*«»\"") if n else "")

    def orientar(self, lado_a: List[str], lado_b: List[str]) -> Optional[float]:
        """P(lado_a rechaza): 1.0 / 0.0 si las dos posiciones coinciden, 0.5 si se contradicen, None si falla.
        Deja en self.ultima_narrativa la narrativa que escribió el modelo (sirve de título)."""
        ej = self._bloque_ejemplos()
        r1, n1 = self._rechaza(self.chat(P_ORIENTAR_V2.format(ejemplos=ej, a=self._lista(lado_a),
                                                              b=self._lista(lado_b)), 1000))
        r2, n2 = self._rechaza(self.chat(P_ORIENTAR_V2.format(ejemplos=ej, a=self._lista(lado_b),
                                                              b=self._lista(lado_a)), 1000))
        self.ultima_narrativa = n1 or n2
        if r1 is None or r2 is None:
            return None
        votos = [1.0 if r1 == "A" else 0.0, 1.0 if r2 == "B" else 0.0]
        return sum(votos) / 2

    def titular(self, ejemplos: List[str]) -> Optional[dict]:
        t = self.chat(P_TITULO.format(a=self._lista(ejemplos)), 1000)
        if not t:
            return None
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            log.warning("titular: respuesta sin JSON: %r", t[:160])
            return None
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            log.warning("titular: JSON inválido (%s): %r", e, t[:160])
            return None
        titulo = str(d.get("titulo", "")).strip().replace("\n", " ")[:140]
        frases = [str(x).strip().replace("|", " ") for x in (d.get("frases_clave") or []) if str(x).strip()][:6]
        return {"titulo": titulo, "frases_clave": frases} if titulo else None
