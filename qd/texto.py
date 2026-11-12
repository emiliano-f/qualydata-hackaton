"""Normalización de texto, hashes estables y léxicos semilla.

Todo es determinista y no depende de la versión de DuckDB ni de Python
(los hashes usan blake2b, no hash() de Python ni de DuckDB).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para",
         "tdo": "todo", "pq": "porque", "d": "de", "k": "que", "bn": "bien"}

STOPWORDS = set("""a al algo con como de del el ella en es esa ese eso esta este esto la las le lo los
mas me mi muy no nos o para pero por que se si sin su sus te todo todos un una uno y ya ahi aca hay
""".split())

# Semillas: se amplían en H3 con herramientas/entrenar.py (n-gramas que aparecen en >=3 narrativas)
# Observado en práctica (respuestas cortas que acompañan al padre); se matchean con tolerancia a typos
ACUERDO = ["tal cual", "exacto", "totalmente", "coincido", "asi es", "es asi", "ni mas ni menos",
           "lo mismo digo", "100 de acuerdo", "de acuerdo", "totalmente de acuerdo", "coincido totalmente",
           "tremendo pero cierto", "clarisimo", "eso mismo pienso", "eso", "posta", "obvio", "ni hablar"]
DESACUERDO = ["mentira", "falso", "fake", "verso", "chamuyo", "nada que ver", "no es asi", "cualquiera",
              "inventan", "desmintio", "sin pruebas", "trucho", "no es cierto", "dejen de", "no hubo",
              "nadie va a"]

# Negación genérica: solo para ORIENTAR bandos agregando miles de comentarios, no por comentario
NEGACION = ["no", "falso", "mentira", "nada que ver", "verso", "fake", "nadie", "nunca", "jamas", "tampoco",
            "sin pruebas", "desmintio", "chamuyo", "trucho", "inventan", "cualquiera"]

_RE_ENTIDADES = re.compile(r"@\w+|#\w+|https?://\S+|www\.\S+")
_RE_NO_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_REPETIDAS = re.compile(r"(\w)\1{2,}")
_RE_ESPACIOS = re.compile(r"\s+")


def normalizar(texto) -> str:
    """minúsculas, sin acentos, sin @/#/URLs, sin símbolos, letras repetidas colapsadas, abreviaturas expandidas."""
    if not isinstance(texto, str) or not texto:
        return ""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = _RE_ENTIDADES.sub(" ", t)
    t = _RE_NO_ALNUM.sub(" ", t)
    t = _RE_REPETIDAS.sub(r"\1", t)
    return " ".join(ABREV.get(w, w) for w in _RE_ESPACIOS.split(t) if w)


def esqueleto(norm: str) -> str:
    """Tokens de 3+ letras sin stopwords, únicos y ordenados: invariante al reordenamiento y a muletillas."""
    return " ".join(sorted({w for w in norm.split() if len(w) >= 3 and w not in STOPWORDS}))


def hash64(s: str) -> int:
    """Hash estable de 63 bits (entra en BIGINT con signo de DuckDB/pandas sin overflow)."""
    if not s:
        return 0
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") >> 1


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
