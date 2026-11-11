"""Genera un set de juguete con el mismo formato que datos/practica (incluida verdad/).

Sirve para `make humo`: probar el pipeline de punta a punta en segundos, sin el set real.
Trae 3 narrativas (una emergente), ruido, respuestas cortas, 1 campaña copia-pega y 1 momento viral.
"""
import argparse
import gzip
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

TZ = timezone(timedelta(hours=-3))
NARR = {
    "nA": ("la obra esta parada porque se robaron la plata",
           ["se robaron la plata de la obra", "la obra no avanza porque se afanaron todo", "donde esta la plata de la obra"],
           ["es mentira que se robaron la plata", "la obra esta parada por la lluvia no por robo", "nada que ver, no hubo robo"]),
    "nB": ("el intendente prometio cloacas y no cumplio",
           ["prometieron cloacas y nada", "las cloacas eran para marzo y seguimos igual", "otra promesa de cloacas incumplida"],
           ["las cloacas ya empezaron en el barrio sur", "falso, las cloacas estan en obra", "dejen de mentir, ya hay cloacas"]),
    "nC": ("van a cerrar el hospital regional",
           ["cierran el hospital regional", "dicen que el hospital cierra en diciembre", "nos dejan sin hospital"],
           ["el hospital no cierra, lo desmintio el ministerio", "verso, el hospital sigue abierto", "sin pruebas de que cierre el hospital"]),
}
RUIDO = ["jajaja", "buen dia", "@marta mira esto", "que calor hoy", "primero!!", "alguien sabe a que hora juega el club"]
CORTAS = ["Tal cual", "Exacto", "Totalmente"]


def iso(dt):
    return dt.astimezone(TZ).strftime("%Y-%m-%dT%H:%M:%S-03:00")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", type=Path, default=Path("datos/juguete"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--semilla", type=int, default=7)
    a = ap.parse_args()
    rnd = random.Random(a.semilla)
    t0 = datetime(2026, 9, 1, 8, 0, tzinfo=TZ)
    dias = 4
    (a.destino / "verdad").mkdir(parents=True, exist_ok=True)

    fuentes = pd.DataFrame({"fuente_id": ["f1", "f2", "f3"], "nombre": ["Diario", "Radio", "Vecinos"],
                            "tipo": ["medio", "radio", "comunidad"]})
    pubs = pd.DataFrame([{"publicacion_id": f"p{i:03d}", "fuente_id": rnd.choice(fuentes.fuente_id),
                          "fecha": iso(t0 + timedelta(hours=2 * i)), "titular": f"Titular {i}"} for i in range(40)])
    genuinas = [f"u{i:04d}" for i in range(300)]
    campana = [f"u{i:04d}" for i in range(300, 320)]
    cuentas = [{"cuenta_id": c, "usuario": f"user_{c}", "seguidores": rnd.randint(10, 3000), "siguiendo": rnd.randint(10, 800),
                "fecha_creacion": iso(t0 - timedelta(days=rnd.randint(200, 3000)))} for c in genuinas]
    lote = t0 - timedelta(days=12)
    cuentas += [{"cuenta_id": c, "usuario": f"patria_{c}", "seguidores": rnd.randint(0, 15), "siguiendo": rnd.randint(200, 900),
                 "fecha_creacion": iso(lote + timedelta(minutes=rnd.randint(0, 50)))} for c in campana]

    filas, verdad = [], []
    emergente_t = t0 + timedelta(days=2, hours=3)

    def agregar(cid, pub, cta, dt, texto, resp, narr, post):
        filas.append({"comentario_id": cid, "publicacion_id": pub, "cuenta_id": cta, "fecha": iso(dt), "texto": texto,
                      "respuesta_a": resp, "me_gusta": rnd.randint(0, 40)})
        verdad.append({"comentario_id": cid, "narrativa_id": narr, "postura": post})

    for i in range(a.n):
        cid = f"c{i:06d}"
        dt = t0 + timedelta(seconds=rnd.randint(0, dias * 86400))
        pub = rnd.choice(pubs.publicacion_id)
        cta = rnd.choice(genuinas)
        if rnd.random() < 0.15:
            agregar(cid, pub, cta, dt, rnd.choice(RUIDO), "", "ninguna", "")
            continue
        narr = rnd.choice(["nA", "nB"] + (["nC"] if dt >= emergente_t else []))
        post = "rechaza" if rnd.random() < 0.3 else "apoya"
        texto = rnd.choice(NARR[narr][1 if post == "apoya" else 2])
        if rnd.random() < 0.3:
            texto = texto.capitalize() + rnd.choice(["", "!!", " 😡", " q verguenza"])
        agregar(cid, pub, cta, dt, texto, "", narr, post)
        if rnd.random() < 0.15:  # respuesta corta que hereda
            agregar(cid + "r", pub, rnd.choice(genuinas), dt + timedelta(minutes=rnd.randint(1, 90)),
                    rnd.choice(CORTAS), cid, narr, post)
    # Primer comentario real de la emergente (define su origen)
    agregar("c_origen_nC", "p020", "u0001", emergente_t, NARR["nC"][1][0], "", "nC", "apoya")

    # Campaña copia-pega: mismas 20 cuentas, 8 publicaciones, ráfagas de <2 min
    for j, pub in enumerate(rnd.sample(list(pubs.publicacion_id), 8)):
        base = t0 + timedelta(hours=6 + 10 * j)
        for c in campana:
            agregar(f"k{j}_{c}", pub, c, base + timedelta(seconds=rnd.randint(0, 100)),
                    "Se robaron la plata de la obra, que devuelvan todo", "", "nA", "apoya")
    # Momento viral genuino: 120 cuentas distintas en 10 minutos sobre una sola publicación
    vt = t0 + timedelta(days=1, hours=20)
    for j, c in enumerate(rnd.sample(genuinas, 120)):
        agregar(f"v{j:03d}", "p010", c, vt + timedelta(seconds=rnd.randint(0, 600)),
                rnd.choice(NARR["nB"][1]), "", "nB", "apoya")

    com = pd.DataFrame(filas)
    with gzip.open(a.destino / "comentarios.csv.gz", "wt", encoding="utf-8", newline="") as f:
        com.to_csv(f, index=False)
    with gzip.open(a.destino / "publicaciones.csv.gz", "wt", encoding="utf-8", newline="") as f:
        pubs.to_csv(f, index=False)
    with gzip.open(a.destino / "cuentas.csv.gz", "wt", encoding="utf-8", newline="") as f:
        pd.DataFrame(cuentas).to_csv(f, index=False)
    fuentes.to_csv(a.destino / "fuentes.csv", index=False)

    pd.DataFrame(verdad).to_csv(a.destino / "verdad" / "comentarios.csv", index=False)
    pd.DataFrame([{"cuenta_id": c["cuenta_id"], "campana_id": "camp1" if c["cuenta_id"] in campana else ""}
                  for c in cuentas]).to_csv(a.destino / "verdad" / "cuentas.csv", index=False)
    pd.DataFrame([{"narrativa_id": k, "descripcion": v[0], "emergente": "si" if k == "nC" else "no",
                   "fecha_origen": iso(emergente_t if k == "nC" else t0)} for k, v in NARR.items()]
                 ).to_csv(a.destino / "verdad" / "narrativas.csv", index=False)
    print(f"juguete: {len(com)} comentarios, {len(cuentas)} cuentas -> {a.destino}")


if __name__ == "__main__":
    main()
