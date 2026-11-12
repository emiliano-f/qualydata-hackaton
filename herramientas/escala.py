"""escala.py — replica un set k veces para medir costo/escala sin verdad.

Cada réplica r0_, r1_, ... tiene ids prefijados (comentario/publicacion/cuenta) y sus fechas
desplazadas k * (span + 1 día), así no se cruzan ventanas ni comunidades entre réplicas.
respuesta_a se prefija igual que comentario_id (o queda vacía). No se copia verdad/.

Uso: python herramientas/escala.py --origen datos/practica --destino work/datos/escala_x5 --factor 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", type=Path, required=True)
    ap.add_argument("--destino", type=Path, required=True)
    ap.add_argument("--factor", type=int, required=True)
    a = ap.parse_args()

    com = pd.read_csv(a.origen / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    pub = pd.read_csv(a.origen / "publicaciones.csv.gz", dtype=str, keep_default_na=False)
    cta = pd.read_csv(a.origen / "cuentas.csv.gz", dtype=str, keep_default_na=False)

    t0 = pd.to_datetime(com["fecha"]).min()
    span = pd.to_datetime(com["fecha"]).max() - t0 + pd.Timedelta(days=1)

    coms, pubs, ctas = [], [], []
    for k in range(a.factor):
        c = com.copy()
        c["comentario_id"] = f"r{k}_" + c["comentario_id"]
        c["publicacion_id"] = f"r{k}_" + c["publicacion_id"]
        c["cuenta_id"] = f"r{k}_" + c["cuenta_id"]
        c["respuesta_a"] = c["respuesta_a"].map(lambda x: f"r{k}_{x}" if x else "")
        c["fecha"] = (pd.to_datetime(c["fecha"]) + k * span).dt.strftime("%Y-%m-%dT%H:%M:%S-03:00")
        p = pub.copy()
        p["publicacion_id"] = f"r{k}_" + p["publicacion_id"]
        p["fecha"] = (pd.to_datetime(p["fecha"]) + k * span).dt.strftime("%Y-%m-%dT%H:%M:%S-03:00")
        u = cta.copy()
        u["cuenta_id"] = f"r{k}_" + u["cuenta_id"]
        u["usuario"] = f"r{k}_" + u["usuario"]
        coms.append(c)
        pubs.append(p)
        ctas.append(u)

    a.destino.mkdir(parents=True, exist_ok=True)
    pd.concat(coms).to_csv(a.destino / "comentarios.csv.gz", index=False, compression="gzip")
    pd.concat(pubs).to_csv(a.destino / "publicaciones.csv.gz", index=False, compression="gzip")
    pd.concat(ctas).to_csv(a.destino / "cuentas.csv.gz", index=False, compression="gzip")
    fuentes = a.origen / "fuentes.csv"
    if fuentes.exists():
        (a.destino / "fuentes.csv").write_bytes(fuentes.read_bytes())
    print(f"{a.destino}: {a.factor} réplicas, {len(com) * a.factor} comentarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
