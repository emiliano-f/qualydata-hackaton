"""Diagnóstico de una corrida contra la verdad, usando las tablas intermedias de work/<set>/qd.duckdb.

Uso: python herramientas/diagnostico.py --datos datos/practica
Muestra: narrativas predichas vs reales, postura por fuente y por narrativa, si la orientación de
cada bando fue correcta (y qué voto acertó), y grupos de coordinación vs campañas.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def leer_tablas(work: Path, nombres):
    import duckdb

    con = duckdb.connect(str(work / "qd.duckdb"), read_only=True)
    existentes = set(con.execute("SELECT table_name FROM information_schema.tables").df()["table_name"])
    out = {n: con.execute(f'SELECT * FROM "{n}"').df() for n in nombres if n in existentes}
    con.close()
    return out


def leer_verdad(v: Path, nombre: str) -> pd.DataFrame:
    from qd.entrada import ubicar

    r = ubicar(v, nombre)
    return pd.read_csv(r, dtype=str, keep_default_na=False) if r else pd.DataFrame()


def diagnosticar(t: dict, v: Path) -> None:
    pd.set_option("display.width", 200)
    vc = leer_verdad(v, "comentarios.csv")
    vk = leer_verdad(v, "cuentas.csv")
    a = t["asig"].merge(vc.rename(columns={"postura": "postura_real"}), on="comentario_id")

    print("## Narrativas predichas (filas) vs reales (columnas), top 25 por tamaño")
    ct = pd.crosstab(a["narrativa"].replace("", "(ruido)"), a["narrativa_id"])
    print(ct.loc[ct.sum(axis=1).sort_values(ascending=False).index[:25]].to_string())

    nar = a[a["narrativa_id"] != "ninguna"]
    nar = nar.assign(ok=nar["postura"] == nar["postura_real"])
    print("\n## Postura: acierto por fuente")
    print(nar.groupby("fuente_postura")["ok"].agg(acierto="mean", n="size").round(3).to_string())
    print("\n## Postura: acierto por narrativa REAL")
    print(nar.groupby("narrativa_id")["ok"].agg(acierto="mean", n="size").round(3).to_string())

    if "bandos_diag" in t and len(t["bandos_diag"]) and "micro" in a.columns:
        print("\n## Orientación de bandos (¿el lado marcado 'rechaza' rechaza de verdad?)")
        b = nar[nar["fuente_postura"] == "bando"]
        acc = b.groupby("narrativa")["ok"].mean().rename("acierto_bando")
        d = t["bandos_diag"].merge(acc, left_on="narrativa", right_index=True, how="left")
        d["orientacion"] = d["acierto_bando"].map(lambda x: "OK" if x >= 0.5 else "INVERTIDA" if x == x else "-")
        print(d.round(3).to_string(index=False))
        print("  Si hay INVERTIDAS: mirá qué voto (neg / coord / dt_h) apuntaba bien y ajustá postura.voto_*")

    if "grupos" in t and len(vk):
        g = t["grupos"][t["grupos"]["cuenta_id"] != ""].merge(vk, on="cuenta_id", how="left")
        print("\n## Grupos predichos vs campañas reales")
        if len(g):
            print(pd.crosstab(g["grupo"], g["campana_id"].replace("", "(genuina)")).to_string())
        faltan = vk[(vk["campana_id"] != "") & ~vk["cuenta_id"].isin(g["cuenta_id"])]
        print(f"- cuentas de campaña no detectadas: {len(faltan)} {faltan['campana_id'].value_counts().to_dict()}")
    if "coord_pares_todos" in t and len(vk):
        cp = t["coord_pares_todos"]
        camp = vk.set_index("cuenta_id")["campana_id"]
        cu, cv = camp.reindex(cp["u"]).fillna("").to_numpy(), camp.reindex(cp["v"]).fillna("").to_numpy()
        tipo = np.where((cu != "") & (cu == cv), cu, np.where((cu == "") & (cv == ""), "genuino", "mixto"))
        print("\n## Pares con n_pub>=min_pub por tipo (para calibrar lift_min / soporte_min)")
        print(cp.assign(tipo=tipo).groupby("tipo")[["n_pub", "score", "esperado", "lift"]]
              .quantile([0.1, 0.5, 0.9]).round(2).unstack().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=Path("datos/practica"))
    ap.add_argument("--work", type=Path, default=None)
    a = ap.parse_args()
    work = a.work or Path("work") / a.datos.name
    t = leer_tablas(work, ["asig", "bandos_diag", "grupos", "coord_pares_todos"])
    if "asig" not in t:
        sys.exit(f"No hay tabla asig en {work}/qd.duckdb: correr make run primero")
    diagnosticar(t, a.datos / "verdad")


if __name__ == "__main__":
    main()
