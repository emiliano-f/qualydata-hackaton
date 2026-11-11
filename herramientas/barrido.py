"""Barrido de un parámetro contra el puntaje oficial, reutilizando las tablas en caché.

Uso:
  python herramientas/barrido.py --param narr.q_T --valores 0.3,0.4,0.5,0.6,0.7,0.8 --desde e04
  python herramientas/barrido.py --param origen.k --valores 1,2,3,5,8 --desde e07 --fijo narr.q_T=0.5
Requisito: una corrida completa previa (make run) para que existan las tablas de work/<set>/.
Cada corrida se agrega a experimentos.csv.
"""
import argparse
import csv
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "evaluacion"))

from qd import pipeline  # noqa: E402
from qd.config import aplicar_sets, cargar, huella  # noqa: E402
import puntuar  # noqa: E402


def puntuar_salida(verdad: Path, salida: Path) -> dict:
    with redirect_stdout(io.StringIO()):
        n = puntuar.score_narratives(verdad, salida)
        c = puntuar.score_coordination(verdad, salida)
    return {"total": round(n["puntos"] + c["puntos"], 2), "narr": n["puntos"], "coord": c["puntos"],
            "bcubed": round(n["bcubed_f1"], 4), "bP": round(n["bcubed_precision"], 4),
            "bR": round(n["bcubed_recall"], 4), "ruido": round(n["ruido_f1"], 4),
            "postura": round(n["postura_macro_f1"], 4), "origen": round(n["origen"], 4),
            "coord_f1": round(c["f1"], 4), "grupos": round(c["grupos_bcubed_f1"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"))
    ap.add_argument("--param", required=True)
    ap.add_argument("--valores", required=True, help="separados por coma")
    ap.add_argument("--desde", default="e04")
    ap.add_argument("--fijo", action="append", default=[], help="clave=valor fijo para todas las corridas")
    ap.add_argument("--registro", type=Path, default=Path("experimentos.csv"))
    a = ap.parse_args()

    filas = []
    for val in [x.strip() for x in a.valores.split(",") if x.strip()]:
        sets = a.fijo + [f"{a.param}={val}"]
        salida = Path("salida") / "barrido" / f"{a.param}={val}"
        t = time.perf_counter()
        rc = pipeline.main(["--datos", str(a.datos), "--salida", str(salida), "--desde", a.desde,
                            "--no-validar", "--sin-llm"] + sum([["--set", s] for s in sets], []))
        if rc != 0:
            print(f"{a.param}={val}: el pipeline falló (rc={rc}), ver {salida}/pipeline.log")
            continue
        r = puntuar_salida(a.datos / "verdad", salida)
        r.update({"param": a.param, "valor": val, "seg": round(time.perf_counter() - t, 1)})
        filas.append(r)
        cfg = aplicar_sets(cargar(Path("config.yaml")), sets)
        nuevo = not a.registro.exists()
        with open(a.registro, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ts", "config", "sets", "datos"] + list(r))
            if nuevo:
                w.writeheader()
            w.writerow({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": huella(cfg), "sets": ";".join(sets),
                        "datos": str(a.datos), **r})

    if not filas:
        return 1
    cols = ["valor", "total", "narr", "coord", "bcubed", "bP", "bR", "ruido", "postura", "origen", "seg"]
    print(f"\nBarrido de {a.param} (fijos: {a.fijo or '-'})")
    print("  ".join(f"{c:>8}" for c in cols))
    mejor = max(filas, key=lambda r: r["total"])
    for r in filas:
        print("  ".join(f"{str(r[c]):>8}" for c in cols) + ("   <- mejor" if r is mejor else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
