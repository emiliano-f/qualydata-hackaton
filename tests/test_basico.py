"""Tests rápidos sin datos reales: python -m pytest tests -q  (o python tests/test_basico.py)."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd import entrada  # noqa: E402
from qd.etapas import e01_ingesta, e09_exporta  # noqa: E402
from qd.texto import esqueleto, hash64, normalizar  # noqa: E402


def test_normalizar():
    assert normalizar("¡¡Se ROBARON la plataaaa!! @juan https://x.co #obra") == "se robaron la plata"
    assert normalizar("xq tmb q") == "porque tambien que"
    assert normalizar(None) == "" and normalizar("") == ""
    assert esqueleto("la plata se robaron") == esqueleto("se robaron la plata")
    assert hash64("a") == hash64("a") and 0 <= hash64("a") < 2**63 and hash64("") == 0


def test_epoch_formatos():
    s = pd.Series(["2026-09-07T08:15:00-03:00", "2026-09-07T11:15:00Z", "basura", ""])
    e = entrada.a_epoch(s)
    assert e[0] == e[1] and pd.isna(e[2]) and pd.isna(e[3])


def test_iso_ar_roundtrip():
    ts = int(datetime.fromisoformat("2026-09-07T08:15:42-03:00").timestamp())
    s = e09_exporta.iso_ar(ts)
    assert s == "2026-09-07T08:15:00-03:00"
    assert datetime.fromisoformat(s).tzinfo is not None


def test_construir_com():
    raw = pd.DataFrame({
        "comentario_id": ["c1", "c2", "c2", "c3", ""],
        "publicacion_id": ["p"] * 5, "cuenta_id": ["u1", "u2", "u2", "u3", "u4"],
        "fecha": ["2026-09-07T08:00:00-03:00", "2026-09-07T09:00:00-03:00", "2026-09-07T09:00:00-03:00",
                  "malísima", "2026-09-07T09:00:00-03:00"],
        "texto": ["Hola", "Tal cual", "dup", "x", "y"], "respuesta_a": ["", "c1", "", "zzz", ""],
        "me_gusta": ["3", "abc", "1", "", "0"]})
    ids, com, desc = e01_ingesta.construir_com(raw)
    assert sorted(ids.comentario_id) == ["c1", "c2", "c3"]
    assert sorted(com.comentario_id) == ["c1", "c2"]
    assert set(desc.motivo) == {"id_duplicado", "fecha_invalida"}
    assert com.set_index("comentario_id").loc["c2", "respuesta_a"] == "c1"
    assert com.set_index("comentario_id").loc["c2", "me_gusta"] == 0


def test_exporta_blindajes():
    ids = pd.DataFrame({"comentario_id": ["c1", "c2", "c3", "c4"]})
    com = pd.DataFrame({"comentario_id": ["c1", "c2", "c3"], "ts": [1000, 2000, 3000], "cuenta_id": ["u1", "u2", "u3"]})
    asig = pd.DataFrame({"comentario_id": ["c1", "c2", "c3"], "narrativa": ["n1", "n2", "ninguna"],
                         "postura": ["rechaza", None, "nan"]})
    narr = pd.DataFrame({"narrativa": ["n1"], "titulo": ["t\nuno"], "frases_clave": ["a|b"], "ts_origen": [600]})
    c = e09_exporta.armar_comentarios(ids, asig)
    assert list(c.postura) == ["rechaza", "apoya", "apoya", "apoya"]     # nunca vacía
    assert list(c.narrativa) == ["n1", "n2", "", ""]
    n = e09_exporta.armar_narrativas(c, narr, com)
    assert set(n.narrativa) == {"n1", "n2"}                              # n2 completada con respaldo
    assert n.set_index("narrativa").loc["n1", "titulo"] == "t uno"
    assert all(datetime.fromisoformat(f).tzinfo for f in n.fecha_origen)
    cta = pd.DataFrame({"cuenta_id": ["u1", "u9"]})
    grupos = pd.DataFrame({"cuenta_id": ["u1", "u2"], "grupo": ["g001", float("nan")]})
    k = e09_exporta.armar_cuentas(cta, com, grupos).set_index("cuenta_id")
    assert k.loc["u1", "grupo"] == "g001" and k.loc["u2", "grupo"] == "" and k.loc["u9", "grupo"] == ""
    assert set(k.index) == {"u1", "u2", "u3", "u9"}


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("ok", nombre)
