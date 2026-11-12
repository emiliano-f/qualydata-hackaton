"""Arma ejemplos resueltos (few-shot) para orientar bandos, a partir de la VERDAD del set de práctica.

Uso: python herramientas/fewshot.py --datos datos/practica [--n 3] [--por_lado 4]
Escribe qd/fewshot_postura.json (se commitea). El enunciado permite aprender de la verdad de práctica:
los temas del set final cambian, pero la convención de qué es "la narrativa" (apoya) y qué es su
rechazo es del generador y se mantiene. Se elige una narrativa por tema, las más grandes.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qd import entrada  # noqa: E402
from qd.texto import normalizar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--por_lado", type=int, default=4)
    ap.add_argument("--destino", type=Path, default=Path("qd/fewshot_postura.json"))
    a = ap.parse_args()
    v = a.datos / "verdad"
    vn = pd.read_csv(entrada.ubicar(v, "narrativas.csv"), dtype=str, keep_default_na=False)
    vc = pd.read_csv(entrada.ubicar(v, "comentarios.csv"), dtype=str, keep_default_na=False)
    com = entrada.leer(a.datos, "comentarios")
    m = com.merge(vc, on="comentario_id")
    m = m[(m["respuesta_a"] == "") & (m["narrativa_id"] != "ninguna")]
    m["norm"] = m["texto"].map(normalizar)
    m["n_tok"] = m["norm"].str.split().str.len()
    m = m[(m["n_tok"] >= 6) & (m["n_tok"] <= 28) & ~m["texto"].str.contains("#")]
    tam = m["narrativa_id"].value_counts()
    vn["n"] = vn["narrativa_id"].map(tam).fillna(0)
    col_tema = "tema" if "tema" in vn.columns else None
    elegidas = (vn.sort_values("n", ascending=False).drop_duplicates(col_tema) if col_tema else
                vn.sort_values("n", ascending=False)).head(a.n)
    ejemplos = []
    for r in elegidas.itertuples():
        g = m[m["narrativa_id"] == r.narrativa_id].drop_duplicates("norm")
        lado = {p: g[g["postura"] == p].sample(frac=1, random_state=0)["texto"].head(a.por_lado).str.strip().tolist()
                for p in ("apoya", "rechaza")}
        ejemplos.append({"narrativa": getattr(r, "titulo", r.narrativa_id), "apoyan": lado["apoya"],
                         "rechazan": lado["rechaza"]})
    a.destino.write_text(json.dumps(ejemplos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(ejemplos)} ejemplos -> {a.destino}")
    for e in ejemplos:
        print(f"- {e['narrativa']}\n    apoya: {e['apoyan'][0]}\n    rechaza: {e['rechazan'][0]}")


if __name__ == "__main__":
    main()
