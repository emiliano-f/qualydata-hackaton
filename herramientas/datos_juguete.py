"""Genera un set de juguete con el mismo formato y la misma ESTRUCTURA que datos/practica.

Reproduce lo observado en la verdad de práctica (make eda):
- respuestas: siempre de la narrativa del padre; cortas acompañan, largas invierten la postura
- el ruido nunca recibe ni hace respuestas; plantillas de ruido usadas por muchas cuentas
- rechazos como contra-afirmaciones (pocas negaciones explícitas), postura ~50/50
- 1 narrativa emergente originada por una campaña nocturna
- 3 campañas: C1 cuentas nuevas + texto casi idéntico + ráfagas diurnas; C2 cuentas viejas + textos
  variados + mismas publicaciones + se responden entre sí; C3 cuentas nuevas + ráfagas nocturnas
- trampas: una publicación viral y "madrugadores" genuinos que comentan temprano en casi todo
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
           ["se robaron la plata de la obra", "la obra no avanza porque se afanaron todo",
            "donde esta la plata de la obra", "la obra esta frenada por la corrupcion"],
           ["la obra avanza lento pero avanza", "ya retomaron la obra la semana pasada",
            "la obra se freno por la lluvia", "lo del robo es falso"]),
    "nB": ("el intendente prometio cloacas y no cumplio",
           ["prometieron cloacas y nada", "las cloacas eran para marzo y seguimos igual",
            "otra promesa de cloacas incumplida", "seguimos sin cloacas en el barrio"],
           ["las cloacas ya empezaron en el barrio sur", "una licitacion de cloacas tarda anos",
            "las maquinas de cloacas estan trabajando", "no es cierto que no hicieron nada"]),
    "nC": ("van a cerrar el hospital regional",
           ["cierran el hospital regional", "dicen que el hospital cierra en diciembre",
            "nos dejan sin hospital regional"],
           ["el hospital sigue abierto y atendiendo", "este ano nombraron medicos nuevos en el hospital",
            "el ministerio confirmo que el hospital sigue"]),
}
RUIDO = ["en que calle queda la obra", "alguien sabe si manana hay paro", "a que hora pasa la ultima",
         "hermoso dia para estar en casa", "jajaja", "buen dia", "@marta mira esto", "primero!!"]
ACUERDO_R = ["Tal cual", "Exacto", "Totalmente", "Ni mas ni menos", "Lo mismo digo", "Coincido", "Exatco",
             "Asi es", "100 de acuerdo"]
MULETILLAS = ["", "", "", "che ", "la verdad ", "mira ", "jaja "]


def iso(dt):
    return dt.astimezone(TZ).strftime("%Y-%m-%dT%H:%M:%S-03:00")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", type=Path, default=Path("work/datos/juguete"))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--semilla", type=int, default=7)
    a = ap.parse_args()
    rnd = random.Random(a.semilla)
    t0 = datetime(2026, 9, 1, 8, 0, tzinfo=TZ)
    dias = 6
    t_fin = t0 + timedelta(days=dias)
    emergente_t = t0 + timedelta(days=3, hours=-6)  # 02:00, ráfaga nocturna de C3
    (a.destino / "verdad").mkdir(parents=True, exist_ok=True)

    fuentes = pd.DataFrame({"fuente_id": ["f1", "f2", "f3"], "nombre": ["Diario", "Radio", "Vecinos"],
                            "tipo": ["medio", "radio", "comunidad"]})
    pubs = [{"publicacion_id": f"p{i:03d}", "fuente_id": rnd.choice(list(fuentes.fuente_id)),
             "t": t0 + timedelta(hours=2 * i), "titular": f"Titular {i}"} for i in range(dias * 12 - 6)]
    pub_t = {p["publicacion_id"]: p["t"] for p in pubs}

    cuentas, camp_de = [], {}

    def nueva_cuenta(cid, creada, seg, usuario):
        cuentas.append({"cuenta_id": cid, "usuario": usuario, "fecha_creacion": iso(creada),
                        "seguidores": seg, "siguiendo": rnd.randint(20, 900)})

    genuinas = [f"u{i:04d}" for i in range(800)]
    for c in genuinas:
        nueva_cuenta(c, t0 - timedelta(days=rnd.randint(300, 4000)), rnd.randint(5, 3000), f"user_{c}")
    madrugadores = genuinas[:30]
    c1 = [f"a{i:03d}" for i in range(20)]
    c2 = [f"b{i:03d}" for i in range(10)]
    c3 = [f"d{i:03d}" for i in range(8)]
    for c in c1:
        nueva_cuenta(c, t0 - timedelta(days=4, minutes=rnd.randint(0, 1400)), rnd.randint(0, 30), f"si_litio_{c}")
        camp_de[c] = "C1"
    for c in c2:
        nueva_cuenta(c, t0 - timedelta(days=rnd.randint(500, 3000)), rnd.randint(30, 300), f"vecino_{c}")
        camp_de[c] = "C2"
    for c in c3:
        nueva_cuenta(c, emergente_t - timedelta(days=1, minutes=rnd.randint(0, 600)), rnd.randint(0, 20), f"info_{c}")
        camp_de[c] = "C3"

    filas, verdad = [], []

    def agregar(cid, pub, cta, dt, texto, resp, narr, post):
        dt = min(dt, t_fin - timedelta(seconds=1))
        filas.append({"comentario_id": cid, "publicacion_id": pub, "cuenta_id": cta, "fecha": dt,
                      "texto": texto, "respuesta_a": resp, "me_gusta": rnd.randint(0, 40)})
        verdad.append({"comentario_id": cid, "narrativa_id": narr, "postura": post})

    def narrativas_en(dt):
        return ["nA", "nB"] + (["nC"] if dt >= emergente_t else [])

    def texto_de(narr, post):
        base = rnd.choice(NARR[narr][1 if post == "apoya" else 2])
        return rnd.choice(MULETILLAS) + base + rnd.choice(["", "", "!!", " 😡", " q verguenza"])

    def momento(pub):
        return pub_t[pub] + timedelta(minutes=rnd.expovariate(1 / 240))

    # 1. Comentarios genuinos
    for i in range(a.n):
        pub = rnd.choice(pubs)["publicacion_id"]
        dt = momento(pub)
        cta = rnd.choice(genuinas)
        if rnd.random() < 0.18:
            agregar(f"c{i:06d}", pub, cta, dt, rnd.choice(RUIDO), "", "ninguna", "")
            continue
        narr = rnd.choice(narrativas_en(dt))
        post = rnd.choice(["apoya", "rechaza"])
        agregar(f"c{i:06d}", pub, cta, dt, texto_de(narr, post), "", narr, post)

    # 2. Trampa: madrugadores genuinos (comentan en los primeros 20 min de casi todo, sin coordinarse)
    j = 0
    for p in pubs:
        for c in madrugadores:
            if rnd.random() < 0.7:
                dt = p["t"] + timedelta(seconds=rnd.randint(0, 1200))
                narr, post = rnd.choice(narrativas_en(dt)), rnd.choice(["apoya", "rechaza"])
                agregar(f"m{j:05d}", p["publicacion_id"], c, dt, texto_de(narr, post), "", narr, post)
                j += 1
    # 3. Trampa: publicación viral genuina
    vt = pub_t["p010"] + timedelta(minutes=5)
    for j, c in enumerate(rnd.sample(genuinas, 250)):
        agregar(f"v{j:03d}", "p010", c, vt + timedelta(seconds=rnd.randint(0, 600)),
                texto_de("nB", "apoya"), "", "nB", "apoya")

    # 4. Campañas
    diurnas = [p for p in pubs if 10 <= p["t"].hour <= 17]
    for j, p in enumerate(rnd.sample(diurnas, 6)):            # C1: ráfagas de 12 min, texto casi idéntico
        base = p["t"] + timedelta(minutes=rnd.randint(5, 60))
        for c in c1:
            agregar(f"k1_{j}_{c}", p["publicacion_id"], c, base + timedelta(seconds=rnd.randint(0, 720)),
                    rnd.choice(["#ObraYa ", "#ObraYa  "]) + "Se robaron la plata de la obra, que devuelvan todo" +
                    rnd.choice(["", "!", "!!"]), "", "nA", "apoya")
    c2_com = []
    for j, p in enumerate(rnd.sample(pubs, 15)):              # C2: mismas publicaciones, textos variados
        base = p["t"] + timedelta(minutes=rnd.randint(10, 300))
        for c in rnd.sample(c2, 8):
            cid = f"k2_{j}_{c}"
            dt = base + timedelta(seconds=rnd.randint(0, 480))
            previos = [x for x in c2_com if x[1] == p["publicacion_id"] and x[2] != c]
            if previos and rnd.random() < 0.35:
                padre = rnd.choice(previos)
                agregar(cid, p["publicacion_id"], c, dt + timedelta(seconds=60), rnd.choice(ACUERDO_R), padre[0], "nB", "apoya")
            else:
                agregar(cid, p["publicacion_id"], c, dt, texto_de("nB", "apoya"), "", "nB", "apoya")
            c2_com.append((cid, p["publicacion_id"], c))
    for j in range(5):                                        # C3: ráfagas nocturnas, originan nC
        dia = emergente_t + timedelta(days=j * 0.6)
        noche = dia.replace(hour=rnd.randint(1, 4), minute=0) if j else emergente_t
        cand = [p for p in pubs if p["t"] <= noche] or pubs
        p = cand[-1]
        for k, c in enumerate(c3):
            dt = noche + timedelta(seconds=(0 if (j == 0 and k == 0) else rnd.randint(30, 900)))
            agregar(f"k3_{j}_{c}", p["publicacion_id"], c, dt,
                    "URGENTE cierran el hospital regional en diciembre, compartan", "", "nC", "apoya")

    # 5. Respuestas (14%): misma narrativa; cortas acompañan, largas invierten
    narrativos = [(f, v) for f, v in zip(filas, verdad) if v["narrativa_id"] != "ninguna"]
    for j in range(int(0.14 * len(filas))):
        f, v = rnd.choice(narrativos)
        dt = f["fecha"] + timedelta(minutes=rnd.randint(1, 180))
        cta = rnd.choice(genuinas)
        if rnd.random() < 0.22:
            agregar(f"r{j:05d}", f["publicacion_id"], cta, dt, rnd.choice(ACUERDO_R), f["comentario_id"],
                    v["narrativa_id"], v["postura"])
        else:
            op = "rechaza" if v["postura"] == "apoya" else "apoya"
            agregar(f"r{j:05d}", f["publicacion_id"], cta, dt, texto_de(v["narrativa_id"], op), f["comentario_id"],
                    v["narrativa_id"], op)

    com = pd.DataFrame(filas)
    com["fecha"] = com["fecha"].map(iso)
    with gzip.open(a.destino / "comentarios.csv.gz", "wt", encoding="utf-8", newline="") as f:
        com.to_csv(f, index=False)
    pubs_df = pd.DataFrame([{"publicacion_id": p["publicacion_id"], "fuente_id": p["fuente_id"],
                             "fecha": iso(p["t"]), "titular": p["titular"]} for p in pubs])
    with gzip.open(a.destino / "publicaciones.csv.gz", "wt", encoding="utf-8", newline="") as f:
        pubs_df.to_csv(f, index=False)
    with gzip.open(a.destino / "cuentas.csv.gz", "wt", encoding="utf-8", newline="") as f:
        pd.DataFrame(cuentas).to_csv(f, index=False)
    fuentes.to_csv(a.destino / "fuentes.csv", index=False)

    pd.DataFrame(verdad).to_csv(a.destino / "verdad" / "comentarios.csv", index=False)
    pd.DataFrame([{"cuenta_id": c["cuenta_id"], "campana_id": camp_de.get(c["cuenta_id"], "")}
                  for c in cuentas]).to_csv(a.destino / "verdad" / "cuentas.csv", index=False)
    pd.DataFrame([{"narrativa_id": k, "descripcion": v[0], "emergente": "si" if k == "nC" else "no",
                   "fecha_origen": iso(emergente_t if k == "nC" else t0)} for k, v in NARR.items()]
                 ).to_csv(a.destino / "verdad" / "narrativas.csv", index=False)
    print(f"juguete: {len(com)} comentarios, {len(cuentas)} cuentas ({len(camp_de)} coordinadas) -> {a.destino}")


if __name__ == "__main__":
    main()
