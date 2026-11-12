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


def test_origen_robusto():
    from qd.etapas.e07_origen import origen_robusto
    import numpy as np
    ts = np.array([0, 50_000, 100_000, 100_100, 100_200, 100_300])   # outlier temprano + ráfaga real
    assert origen_robusto(ts, k=1, m=3, ventana_s=3600) == 100_000
    assert origen_robusto(ts, k=2, m=3, ventana_s=3600) == 100_100
    assert origen_robusto(np.array([5]), k=3, m=3, ventana_s=10) == 5            # sin ráfaga: k-ésimo acotado


def test_herencia_respuestas():
    from qd.etapas.e06_postura import es_acuerdo, heredar_respuestas
    assert es_acuerdo("tal cual", 3) and es_acuerdo("ni mas ni mneos", 3) and es_acuerdo("exatco trabajo y futuro", 3)
    assert not es_acuerdo("la obra avanza lento pero avanza", 3)
    df = pd.DataFrame({"comentario_id": ["p", "a", "b", "c"], "ts": [0, 1, 2, 3],
                       "respuesta_a": ["", "p", "p", "b"], "narrativa": ["n1", "n2", "n3", "n4"],
                       "postura": ["apoya"] * 4, "acuerdo": [False, True, False, True],
                       "es_ruido": [True, False, False, False]})
    out = heredar_respuestas(df).set_index("comentario_id")
    assert list(out.loc[["a", "b", "c"], "narrativa"]) == ["n1", "n1", "n1"]
    assert list(out.loc[["a", "b", "c"], "postura"]) == ["apoya", "rechaza", "rechaza"]   # b invierte, c acompaña a b
    assert not out.loc["p", "es_ruido"]                                                  # nadie le responde al ruido
    assert not out.loc["p", "heredada"] and out.loc["c", "heredada"]


def test_biparticion_y_pares():
    import numpy as np
    from qd.etapas.e06_postura import biparticion
    from qd.etapas.e08_coord import pares
    W = np.array([[0, 5, 0, 4], [5, 0, 3, 0], [0, 3, 0, 6], [4, 0, 6, 0]], dtype=float)  # ciclo par: 0,2 vs 1,3
    x = biparticion(W)
    assert x[0] == x[2] and x[1] == x[3] and x[0] != x[1]
    post = np.array([0, 0, 0, 1, 1]); ts = np.array([0, 100, 1000, 0, 50])
    acc = np.array([0, 1, 2, 0, 1]); txt = np.zeros(5, dtype=np.int64)
    k, n, s, _ = pares(post, ts, acc, txt, delta=300, bonus=1.0, n_acc=3)
    assert list(k) == [1] and list(n) == [2]          # solo (0,1), en 2 publicaciones


def test_ruido_knn():
    import numpy as np
    from qd.etapas.e02b_ruido import votar_knn
    Xd = np.array([[1, 0], [0.98, 0.2], [0, 1], [0.2, 0.98]], dtype=np.float32)
    Xd /= np.linalg.norm(Xd, axis=1, keepdims=True)
    yd = np.array([1.0, 1.0, 0.0, 0.0])                 # dos plantillas de ruido, dos textos de narrativa
    Xu = np.array([[0.99, 0.1], [0.1, 0.99]], dtype=np.float32)
    p = votar_knn(Xu / np.linalg.norm(Xu, axis=1, keepdims=True), Xd, yd, k=2)
    assert p[0] > 0.9 and p[1] < 0.1


def test_orientar_votos():
    import numpy as np
    from qd.etapas.e06_postura import orientar
    votos = {"neg": 1.0, "coord": 2.0, "temprano": 0.0, "coord_rel": 0.3}
    n = 400
    lado = np.r_[np.ones(n), -np.ones(n)]
    neg = np.r_[np.zeros(n), np.ones(n) * (np.arange(n) < 120)]          # lado -1 niega más
    coord = np.r_[np.arange(n) < 40, np.arange(n) < 20]                    # lado +1 tiene el doble de coordinadas
    b = pd.DataFrame({"lado": lado, "neg": neg, "coord": coord, "ts": np.arange(2 * n), "refuta": False})
    rech, det = orientar(b, votos)
    assert rech == -1                                                       # neg y coord coinciden
    b2 = b.assign(neg=np.r_[np.ones(n) * (np.arange(n) < 120), np.zeros(n)])  # neg contradice a coord
    assert orientar(b2, votos)[0] == -1                                     # coord (peso 2, dif. relativa 0.5) gana
    b3 = b2.assign(coord=np.r_[np.arange(n) < 21, np.arange(n) < 20])      # dif. relativa chica: coord no vota
    assert orientar(b3, votos)[0] == 1


def test_llm_parseo():
    from qd.llm import LLM
    assert LLM._letra("A") == "A" and LLM._letra(" b.") == "B" and LLM._letra("Respuesta: B") == "B"
    assert LLM._letra("ninguno") is None and LLM._letra(None) is None
    assert LLM._rechaza("NARRATIVA: la obra está parada\nRECHAZA: B") == ("B", "la obra está parada")
    assert LLM._rechaza("**NARRATIVA:** «x»\n**RECHAZA:** a")[0] == "A"


def test_llm_cache_y_modo():
    """R2/R5: acierto de caché sin clave devuelve el texto y suma el uso guardado; los 4 modo_llm."""
    import hashlib
    import json
    import os
    import tempfile
    from qd.costo import Costo
    from qd.llm import LLM, SISTEMA
    clave = os.environ.pop("LLM_API_KEY", None)
    try:
        cfg = {"llm": {"cache_dir": tempfile.mkdtemp(), "modelo": "m", "reintentos": 0}}
        c = Costo({}, "x")
        c.resolver_modo_llm()
        assert c.modo_llm == "apagado"                          # ningún LLM instanciado
        c = Costo({}, "x")
        llm = LLM(cfg, c, "set")
        assert llm.degradado and c.llm_usado
        prompt = "prompt de prueba"
        h = hashlib.sha256(json.dumps(["m", SISTEMA, prompt, 200, ""],
                                      ensure_ascii=False).encode()).hexdigest()[:32]
        (llm.cache / f"{h}.json").write_text(
            json.dumps({"texto": "OK", "uso": {"prompt_tokens": 10, "completion_tokens": 5,
                                               "total_tokens": 140}}), encoding="utf-8")
        assert llm.chat(prompt) == "OK"
        assert c.aciertos_cache == 1 and c.tok_in == 10 and c.tok_out == 130   # salida = total - prompt
        c.resolver_modo_llm()
        assert c.modo_llm == "desde_cache" and c.llamadas == 0
        os.environ["LLM_API_KEY"] = "falsa"
        c = Costo({}, "x")
        llm = LLM(cfg, c, "set2")
        llm._post = lambda cuerpo: {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
        assert llm.chat("otro prompt") == "x"
        c.resolver_modo_llm()
        assert c.modo_llm == "completo" and c.llamadas == 1 and c.tok_out == 2
        c = Costo({}, "x")
        llm = LLM(cfg, c, "set3")

        def _boom(cuerpo):
            raise ConnectionError("sin red")
        llm._post = _boom
        assert llm.chat("prompt que falla") is None
        c.resolver_modo_llm()
        assert c.modo_llm == "degradado" and c.fallas_llm == 1
        # sin clave + caché vacío: la respuesta no estaba -> degradado (no "desde_cache" de mentira)
        os.environ.pop("LLM_API_KEY")
        c = Costo({}, "x")
        llm = LLM(cfg, c, "set_vacio")
        assert llm.chat("prompt sin cachear") is None
        assert c.faltas_cache == 1
        c.resolver_modo_llm()
        assert c.modo_llm == "degradado"
        assert any("no estaban en caché" in n for n in c.notas)
        # caché parcial (un acierto + una falta) -> degradado igual
        c = Costo({}, "x")
        llm = LLM(cfg, c, "set")
        assert llm.chat(prompt) == "OK"                # archivo escrito arriba
        assert llm.chat("prompt sin cachear") is None
        c.resolver_modo_llm()
        assert c.modo_llm == "degradado" and c.aciertos_cache == 1 and c.faltas_cache == 1
    finally:
        if clave is not None:
            os.environ["LLM_API_KEY"] = clave
        else:
            os.environ.pop("LLM_API_KEY", None)


def test_llm_truncado_y_reasoning():
    """R3: finish_reason=length no se cachea, no se reintenta y cuenta como falla;
    reasoning_effort viaja en el cuerpo solo si está configurado."""
    import logging
    import os
    import tempfile
    from qd.costo import Costo
    from qd.llm import LLM
    clave = os.environ.get("LLM_API_KEY")
    os.environ["LLM_API_KEY"] = "falsa"
    avisos = []
    h = logging.Handler()
    h.emit = lambda r: avisos.append(r.getMessage())
    logging.getLogger("qd").addHandler(h)
    try:
        cfg = {"llm": {"cache_dir": tempfile.mkdtemp(), "modelo": "m", "reintentos": 0,
                       "reasoning_effort": "low"}}
        c = Costo({}, "x")
        llm = LLM(cfg, c, "set")
        vistos = []

        def _post(cuerpo):
            vistos.append(cuerpo)
            return {"choices": [{"message": {"content": '{"titulo": "x", "fra'}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 121}}

        llm._post = _post
        assert llm.chat("p truncado") is None
        assert len(vistos) == 1                            # no reintenta el mismo prompt
        assert vistos[0]["reasoning_effort"] == "low"      # viaja en el cuerpo
        assert not list(llm.cache.glob("*.json"))          # no se cachea
        assert c.fallas_llm == 1 and c.tok_out == 116      # total - prompt: se factura igual
        assert any("truncad" in a for a in avisos)
        llm2 = LLM({"llm": {"cache_dir": tempfile.mkdtemp(), "modelo": "m", "reintentos": 0}},
                  Costo({}, "x"), "s")
        vistos.clear()

        def _post2(cuerpo):
            vistos.append(cuerpo)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

        llm2._post = _post2
        assert llm2.chat("p") == "ok"
        assert "reasoning_effort" not in vistos[0]         # vacío = no se manda
    finally:
        logging.getLogger("qd").removeHandler(h)
        if clave is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = clave


def test_cohesion_filtro():
    """O1: un grupo que co-actúa en ventana tiene exceso de cohesión vs el nulo; uno disperso no."""
    import numpy as np
    from qd.etapas.e08_coord import cohesion
    # pubs 0-2: tres cuentas comentan juntas dentro de ±60s; pubs 3-8: un evento por publicación
    post = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8])
    ts = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 0, 0, 0, 0, 0])
    acc = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])
    g1, g2 = {0, 1, 2}, {3, 4, 5}
    assert cohesion(post, ts, acc, g1, 60) == 1.0        # todos sus eventos tienen compañía
    assert cohesion(post, ts, acc, g2, 60) == 0.0        # ninguno
    # nulo: las etiquetas de g1 caen una por publicación -> exceso 1.0 (entra); g2 queda en 0 (sale)
    acc_nulo = np.array([6, 7, 8, 6, 7, 8, 6, 7, 8, 0, 1, 2, 0, 1, 2])
    exc1 = cohesion(post, ts, acc, g1, 60) - cohesion(post, ts, acc_nulo, g1, 60)
    exc2 = cohesion(post, ts, acc, g2, 60) - cohesion(post, ts, acc_nulo, g2, 60)
    assert exc1 >= 0.2 and exc2 < 0.2


def test_consolidar_y_sets():
    import numpy as np
    from qd.config import aplicar_sets
    from qd.etapas.e04_narrativas import consolidar
    C = np.array([[1, 0], [0.99, 0.141], [0, 1], [0.141, 0.99]], dtype=np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    g, _, _ = consolidar(C, 0.9)
    assert g[0] == g[1] and g[2] == g[3] and g[0] != g[2]
    cfg = aplicar_sets({"narr": {"q_T": 0.6}}, ["narr.q_T=0.4", "llm.activo=false"])
    assert cfg["narr"]["q_T"] == 0.4 and cfg["llm"]["activo"] is False


def test_embed_e5_corta_sin_respaldo():
    """Si el motor pedido es e5 y no carga, corta con RuntimeError; tfidf solo corre si se pide."""
    import tempfile
    from types import SimpleNamespace

    import numpy as np
    from qd.costo import Costo
    from qd.etapas import e02_embed

    class _DB:
        def __init__(self):
            self.tab = {}
        def leer(self, t):
            return pd.DataFrame({"h_texto": np.array([11, 22, 33], np.int64),
                                 "texto": ["hola mundo", "otro texto", "chau chau"],
                                 "norm": ["hola mundo", "otro texto", "chau chau"]})
        def existe(self, t):
            return t in self.tab
        def guardar(self, t, df):
            self.tab[t] = df

    def _ctx(motor, tmp):
        return SimpleNamespace(db=_DB(), work=Path(tmp), cfg={}, embed=motor, costo=Costo({}, "x"))

    llamado = {"e5": 0}
    orig = e02_embed._e5
    def _falla(*a, **k):
        llamado["e5"] += 1
        raise OSError("sin red")
    e02_embed._e5 = _falla
    try:
        try:
            e02_embed.correr(_ctx("e5", tempfile.mkdtemp()))
        except RuntimeError as e:
            assert str(e) == ("e5 no disponible: corré make instalar con internet o usá "
                              "--embed tfidf a propósito")
            assert e.__cause__ is not None
        else:
            raise AssertionError("e5 sin modelo tenía que cortar, no caer a tfidf")
        ctx2 = _ctx("tfidf", tempfile.mkdtemp())
        e02_embed.correr(ctx2)
        assert llamado["e5"] == 1                                    # tfidf no toca _e5
        assert (ctx2.db.tab["textos_emb"]["motor"] == "tfidf").all()
        assert np.load(ctx2.work / "emb.npy").shape[0] == 3
    finally:
        e02_embed._e5 = orig


def test_costo_extras():
    """ram_pico_gb (unidades según SO), almacenamiento_mb (verdad/ excluida) y cuentas derivadas."""
    import sys
    import tempfile
    from types import SimpleNamespace

    from qd import costo as qcosto
    from qd.costo import Costo

    orig_getrusage, orig_platform = qcosto.resource.getrusage, sys.platform
    try:
        sys.platform = "linux"
        qcosto.resource.getrusage = lambda *a, **k: SimpleNamespace(ru_maxrss=2_000_000)
        assert qcosto._ram_pico_gb() == 2.0                       # 2.000.000 KB -> 2 GB
        sys.platform = "darwin"
        qcosto.resource.getrusage = lambda *a, **k: SimpleNamespace(ru_maxrss=2_000_000_000)
        assert qcosto._ram_pico_gb() == 2.0                       # 2e9 bytes -> 2 GB
    finally:
        qcosto.resource.getrusage = orig_getrusage
        sys.platform = orig_platform

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        datos, work = td / "datos", td / "work"
        (datos / "verdad").mkdir(parents=True)
        (work / "parquet" / "dia=1").mkdir(parents=True)
        (datos / "comentarios.csv").write_bytes(b"x" * 2_000_000)
        (datos / "verdad" / "v.csv").write_bytes(b"y" * 9_000_000)
        (work / "qd.duckdb").write_bytes(b"z" * 3_000_000)
        (work / "qd.duckdb.wal").write_bytes(b"z" * 500_000)
        (work / "parquet" / "dia=1" / "p.parquet").write_bytes(b"z" * 4_000_000)
        assert qcosto.medir_almacenamiento(datos, work) == {"duckdb": 3.5, "parquet": 4.0,
                                                          "entrada": 2.0}

    c = Costo({"usd_hora_cpu": 0.72}, "x")
    c.comentarios = 100_000
    c.t0 -= 100.0
    d = c.a_dict()
    for clave in ("comentarios_procesados", "segundos", "tokens_ia", "usd_ia",
                  "usd_por_mil_comentarios", "hardware", "almacenamiento", "notas"):
        assert clave in d                                        # las 8 obligatorias, intactas
    assert d["comentarios_por_segundo"] == round(100_000 / d["segundos"], 1)
    assert d["segundos_por_mil_comentarios"] == round(d["segundos"] / 100_000 * 1000, 4)
    assert d["usd_computo_por_mil"] == round(d["segundos"] * 0.72 / 3600 / 100_000 * 1000, 6)
    assert d["proyeccion_1M_por_dia"]["horas"] == round(d["segundos"] / 100_000 * 1e6 / 3600, 2)
    assert d["ram_pico_gb"] > 0

    c2 = Costo({}, "x")
    c2.comentarios = 10
    d2 = c2.a_dict()
    assert d2["usd_computo_por_mil"] is None                     # sin usd_hora_cpu -> null
    assert d2["almacenamiento_mb"] is None                       # sin medición -> null


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("ok", nombre)
