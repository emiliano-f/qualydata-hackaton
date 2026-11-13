"""e08 · Coordinación: co-acción repetida contra un modelo nulo + Louvain.

Evento = primer comentario de una cuenta en una publicación. Dos cuentas co-actúan cuando comentan la
misma publicación con |Δt| <= coord.delta_s. Peso de cada co-evento:
    w = (1 + bonus_texto·[mismo texto esqueleto, >= min_tok_texto tokens]) / ln(2 + N_ventana)
con N_ventana = cuentas en esa publicación dentro de ±Δ (penaliza momentos virales).
Score del par = suma de w sobre publicaciones distintas; n_pub = cantidad de publicaciones.

Modelo nulo ("reasigna"): cada evento se muda a una publicación al azar (probabilidad ∝ tamaño de la
publicación) conservando su demora desde el inicio de la publicación. Preserva el hábito de cada cuenta
(cuánto comenta, qué tan rápido: los "madrugadores" siguen coincidiendo) y rompe la elección conjunta
de publicaciones, que es lo que delata a una operación. Cada par se compara con SU
esperado: lift = score_obs / (score_nulo_medio + ε), con ε = peso mediano de una co-ocurrencia.
Arista si n_pub >= coord.min_pub y lift >= coord.lift_min. (coord.nulo_reps réplicas del nulo.)

Respuestas repetidas entre dos cuentas (>= coord.min_resp_par) y hashtags poco usados compartidos
(<= coord.hashtag_max_frac de las cuentas) suman como evidencia (n_pub += 1 y score += peso·ε).
Nulo "permuta": lugares (publicación, momento) fijos; se baraja qué cuenta ocupa cada lugar dentro de
fuente x período x banda de demora.
Aristas -> poda por triángulos (vecinos en común / (grado mínimo - 1) >= coord.soporte_min) ->
componentes -> Louvain dentro de cada componente -> grupos de >= coord.min_size con densidad >= coord.densidad_min
y actividad mediana <= percentil coord.max_actividad_q de las cuentas (los hiperactivos coinciden por estar en todo).
Escribe grupos(cuenta_id, grupo) y coord_pares(u, v, n_pub, score).
Implementado en numpy (vectorizado por publicación) para que sea testeable sin SQL; el equivalente
SQL para DuckDB/ClickHouse está en el README (self-join por publicacion_id con rango de ts).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from qd.config import obtener

log = logging.getLogger("qd")
SALIDAS = ["grupos", "coord_pares", "coord_pares_todos", "grupos_diag"]


def eventos(com: pd.DataFrame) -> pd.DataFrame:
    ev = com.sort_values("ts", kind="stable").drop_duplicates(["cuenta_id", "publicacion_id"])
    return ev[["cuenta_id", "publicacion_id", "ts", "h_esq", "n_tok"]].reset_index(drop=True)


def cohesion(post: np.ndarray, ts: np.ndarray, acc: np.ndarray, miembros, delta: int) -> float:
    """Fracción de los eventos de `miembros` (códigos de cuenta) que tienen OTRO evento de un
    miembro en la misma publicación a |Δt| <= delta. Sirve sobre (P,T,A) observadas o sobre las
    réplicas del nulo (P0,T0,A0): una campaña co-actúa; un grupo de hiperactivos genuinos no más
    que por azar. Los eventos se deduplican por (cuenta, publicación) como en `eventos()`."""
    sel = np.isin(acc, np.fromiter(miembros, dtype=acc.dtype))
    if not sel.any():
        return 0.0
    m = pd.DataFrame({"a": acc[sel], "p": post[sel], "t": ts[sel]})
    m = m.drop_duplicates(["a", "p"]).sort_values(["p", "t"], kind="stable")
    pp, tt = m["p"].to_numpy(), m["t"].to_numpy()
    junto = np.zeros(len(m), bool)
    cortes = np.flatnonzero(np.diff(pp)) + 1
    for ini, fin in zip(np.r_[0, cortes], np.r_[cortes, len(pp)]):
        if fin - ini < 2:
            continue
        t = tt[ini:fin]
        junto[ini:fin] = (np.searchsorted(t, t + delta, "right")
                          - np.searchsorted(t, t - delta, "left")) >= 2
    return float(junto.mean())


def cohesion_cruzada(post: np.ndarray, ts: np.ndarray, acc: np.ndarray, ma, mb, delta: int) -> float:
    """Fracción de los eventos de `ma` ∪ `mb` que tienen un evento del OTRO conjunto en la misma
    publicación a |Δt| <= delta. Como `cohesion()` pero mide co-acción ENTRE dos grupos; se aplica
    igual sobre (P,T,A) observadas o sobre las réplicas del nulo."""
    sa = np.isin(acc, np.fromiter(ma, dtype=acc.dtype))
    sb = np.isin(acc, np.fromiter(mb, dtype=acc.dtype))
    if not sa.any() or not sb.any():
        return 0.0
    m = pd.DataFrame({"a": acc[sa | sb], "p": post[sa | sb], "t": ts[sa | sb],
                      "g": np.where(sa[sa | sb], 0, 1)})
    m = m.drop_duplicates(["a", "p"]).sort_values(["p", "t"], kind="stable")
    pp, tt, gg = m["p"].to_numpy(), m["t"].to_numpy(), m["g"].to_numpy()
    junto = np.zeros(len(m), bool)
    cortes = np.flatnonzero(np.diff(pp)) + 1
    for ini, fin in zip(np.r_[0, cortes], np.r_[cortes, len(pp)]):
        if fin - ini < 2:
            continue
        t, g = tt[ini:fin], gg[ini:fin]
        ta, tb = t[g == 0], t[g == 1]
        otro = np.where(g == 0,
                        np.searchsorted(tb, t + delta, "right") - np.searchsorted(tb, t - delta, "left"),
                        np.searchsorted(ta, t + delta, "right") - np.searchsorted(ta, t - delta, "left"))
        junto[ini:fin] = otro >= 1
    return float(junto.mean())


def fusionar_grupos(validas: list, pasa: list) -> dict:
    """Union-find sobre los índices de `validas` unidos por los pares `pasa` (fusión transitiva).
    Devuelve {raiz: set_de_codigos} con la unión de miembros por componente."""
    padre = list(range(len(validas)))

    def find(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i, j in pasa:
        padre[find(j)] = find(i)
    comp: dict = {}
    for i, vc in enumerate(validas):
        comp.setdefault(find(i), set()).update(vc[0])
    return comp


def orden_grupos(comp: dict, cuentas) -> list:
    """Orden determinista para el relabel con fusión: tamaño desc, desempate por menor cuenta_id."""
    return sorted(comp.values(), key=lambda c: (-len(c), min(cuentas[n] for n in c)))


def _frac_casi_identicos(a_df: pd.DataFrame, b_df: pd.DataFrame, fila_de_h, cos_min: float,
                         min_tok: int) -> float:
    """Fracción de filas de a_df con un comentario "casi idéntico" en b_df: mismo h_esq con
    n_tok >= min_tok, o coseno de embeddings >= cos_min."""
    if not len(a_df) or not len(b_df):
        return 0.0
    esq_b = set(b_df.loc[b_df["n_tok"] >= min_tok, "h_esq"])
    ok = (a_df["h_esq"].isin(esq_b) & (a_df["n_tok"] >= min_tok)).to_numpy()
    if fila_de_h is not None:
        sim = fila_de_h(a_df["h_texto"].to_numpy()) @ fila_de_h(b_df["h_texto"].to_numpy()).T
        ok |= (sim >= cos_min).any(axis=1)
    return float(ok.mean())


def pares_fusion(ctx, validas: list, fusion: str, com: pd.DataFrame, cuentas,
                 P: np.ndarray, T: np.ndarray, A: np.ndarray, nulos_pta: list,
                 delta: int, cohesion_min: float) -> list:
    """Pares (i, j) de `validas` que cumplen el criterio `fusion` (v1|v3|v1v3|v2).
    Condición común (a): misma narrativa principal y share >= 0.80 en ambos grupos."""
    cfg = ctx.cfg
    narr = {}
    if ctx.db.existe("asig_narr"):
        an = ctx.db.leer("asig_narr")[["comentario_id", "narrativa", "es_ruido"]]
        an.loc[an["es_ruido"].astype(bool), "narrativa"] = ""
        narr = dict(zip(an["comentario_id"], an["narrativa"]))
    narr_de = com["comentario_id"].map(narr).fillna("")
    coordenadas = set().union(*[set(cuentas[n] for n in c) for c, _, _ in validas])
    info = []
    for c, _, _ in validas:
        sel = com["cuenta_id"].isin({cuentas[n] for n in c})
        labs = narr_de[sel].to_numpy()
        nz = labs[labs != ""]
        main = str(pd.Series(nz).mode().iloc[0]) if len(nz) else ""
        share = float((labs == main).mean()) if len(labs) and main else 0.0
        info.append((main, share))
        log.info("e08: fusión: candidato %d ctas, narrativa principal %r (share %.2f)",
                 len(c), main, share)

    min_tok = int(obtener(cfg, "coord.min_tok_texto", 6))
    cos_min = float(obtener(cfg, "coord.fusion_cos", 0.95))
    frac_min = float(obtener(cfg, "coord.fusion_frac", 0.30))
    ratio_min = float(obtener(cfg, "coord.fusion_ratio", 2.0))
    rng = np.random.default_rng(int(obtener(cfg, "coord.semilla", 42)))

    fila_de_h = None
    if "v3" in fusion:
        try:
            from qd import vectores
            h_txt, X = vectores.cargar(ctx)
            idx = {h: i for i, h in enumerate(h_txt)}

            def fila_de_h(hs):  # filas de emb.npy por h_texto
                return X[np.fromiter((idx[h] for h in hs), dtype=np.int64)]
        except Exception as e:  # noqa: BLE001
            log.warning("e08: fusión v3 sin embeddings (%s); solo cuenta h_esq", e)

    def comentarios_de(c, main):
        sel = com["cuenta_id"].isin({cuentas[n] for n in c}) & (narr_de == main)
        return com.loc[sel, ["h_esq", "n_tok", "h_texto"]]

    pool_no_coord = com.loc[~com["cuenta_id"].isin(coordenadas)]
    pasa = []
    for i in range(len(validas)):
        for j in range(i + 1, len(validas)):
            mi, si = info[i]
            mj, sj = info[j]
            if not (mi and mi == mj and si >= 0.80 and sj >= 0.80):
                continue
            ci, cj = validas[i][0], validas[j][0]
            ok = fusion == "v2"
            det = ""
            if fusion in ("v1", "v1v3"):
                co_x = cohesion_cruzada(P, T, A, ci, cj, delta)
                cn_x = (float(np.mean([cohesion_cruzada(p0, t0, a0, ci, cj, delta)
                                       for p0, t0, a0 in nulos_pta])) if nulos_pta else 0.0)
                ok = ok or (co_x - cn_x >= cohesion_min)
                det = f"v1 coh_x={co_x:.2f} nulo={cn_x:.2f}"
            if not ok and fusion in ("v3", "v1v3"):
                di = comentarios_de(ci, mi)
                dj = comentarios_de(cj, mi)
                pool = pool_no_coord[narr_de[~com["cuenta_id"].isin(coordenadas)] == mi]
                fs, bs = [], []
                for a_df, b_df in ((di, dj), (dj, di)):
                    f = _frac_casi_identicos(a_df, b_df, fila_de_h, cos_min, min_tok)
                    b = float(np.mean([_frac_casi_identicos(
                        a_df, pool.sample(n=min(len(b_df), len(pool)), random_state=rng),
                        fila_de_h, cos_min, min_tok) for _ in range(3)])) if len(pool) else 0.0
                    fs.append(f)
                    bs.append(b)
                ok3 = all(f >= frac_min and f >= ratio_min * b for f, b in zip(fs, bs))
                ok = ok or ok3
                det += f" v3 f={['%.2f' % x for x in fs]} b={['%.2f' % x for x in bs]}"
            if ok:
                pasa.append((i, j))
                log.info("e08: fusión '%s': par %d+%d cumple (%s)%s", fusion, i, j, mi, det)
    return pasa


def pares(post: np.ndarray, ts: np.ndarray, acc: np.ndarray, txt: np.ndarray, delta: int, bonus: float,
          n_acc: int, bloque: int = 5_000_000):
    """Pares co-actuantes, agregados. Entradas ordenadas por (post, ts); cada cuenta tiene un evento por
    publicación. Devuelve (clave_par, n_pub, suma_pesos, mediana_peso). Agrega por bloques de ~`bloque`
    pares para acotar la memoria en sets grandes."""
    claves, pesos, acum, muestras = [], [], [], []
    pend = 0

    def volcar():
        nonlocal claves, pesos, pend
        if claves:
            k, w = np.concatenate(claves), np.concatenate(pesos)
            muestras.append(w[:: max(1, len(w) // 200_000)])
            acum.append(_agregar_crudo(k, w))
        claves, pesos, pend = [], [], 0

    cortes = np.flatnonzero(np.diff(post)) + 1
    for ini, fin in zip(np.r_[0, cortes], np.r_[cortes, len(post)]):
        t, a, x = ts[ini:fin], acc[ini:fin], txt[ini:fin]
        n = len(t)
        if n < 2:
            continue
        izq = np.searchsorted(t, t - delta, side="left")
        der = np.searchsorted(t, t + delta, side="right")
        crowd = der - izq                       # N_ventana (incluye al propio evento)
        cnt = der - np.arange(n) - 1            # vecinos posteriores dentro de Δ
        tot = int(cnt.sum())
        if tot == 0:
            continue
        I = np.repeat(np.arange(n), cnt)
        J = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + I + 1
        w = 1.0 / np.log(2.0 + np.maximum(crowd[I], crowd[J]))
        mismo = (x[I] == x[J]) & (x[I] != 0)
        w = w * (1.0 + bonus * mismo)
        u, v = np.minimum(a[I], a[J]), np.maximum(a[I], a[J])
        claves.append(u.astype(np.int64) * n_acc + v)
        pesos.append(w)
        pend += tot
        if pend >= bloque:
            volcar()
    volcar()
    if not acum:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), 1.0
    k = np.concatenate([x[0] for x in acum])
    n = np.concatenate([x[1] for x in acum])
    sm = np.concatenate([x[2] for x in acum])
    uk, inv = np.unique(k, return_inverse=True)
    return uk, np.bincount(inv, weights=n).astype(np.int64), np.bincount(inv, weights=sm), \
        float(np.median(np.concatenate(muestras)))


def _agregar_crudo(claves: np.ndarray, pesos: np.ndarray):
    uk, inv = np.unique(claves, return_inverse=True)
    return uk, np.bincount(inv), np.bincount(inv, weights=pesos)


def sumar_evidencia(k_obs, n_obs, s_obs, esperado, kr, nr, peso):
    """Suma evidencia extra (respuestas mutuas, hashtags raros) a los pares; agrega pares nuevos si hace falta."""
    todos = np.union1d(k_obs, kr)
    i_o, i_r = np.searchsorted(todos, k_obs), np.searchsorted(todos, kr)
    n2, s2, e2 = np.zeros(len(todos), np.int64), np.zeros(len(todos)), np.zeros(len(todos))
    n2[i_o], s2[i_o], e2[i_o] = n_obs, s_obs, esperado
    np.add.at(n2, i_r, nr)
    np.add.at(s2, i_r, nr * peso)
    return todos, n2, s2, e2


def pares_hashtag(com: pd.DataFrame, cod: pd.Index, n_acc: int, max_ctas: int):
    """Pares de cuentas que comparten un hashtag usado por entre 2 y max_ctas cuentas."""
    h = com[["cuenta_id", "texto"]].assign(tag=com["texto"].str.lower().str.findall(r"#\w{3,}")).explode("tag")
    h = h.dropna(subset=["tag"]).drop_duplicates(["tag", "cuenta_id"])
    if h.empty:
        return np.empty(0, np.int64), np.empty(0, np.int64), 0
    uso = h.groupby("tag")["cuenta_id"].transform("size")
    h = h[(uso >= 2) & (uso <= max_ctas)]
    claves = []
    for _, g in h.groupby("tag"):
        c = np.unique(cod.get_indexer(g["cuenta_id"]))
        c = c[c >= 0]
        if len(c) < 2:
            continue
        i, j = np.triu_indices(len(c), 1)
        claves.append(c[i].astype(np.int64) * n_acc + c[j])
    if not claves:
        return np.empty(0, np.int64), np.empty(0, np.int64), 0
    k, n = np.unique(np.concatenate(claves), return_counts=True)
    return k, n, h["tag"].nunique()


def correr(ctx) -> None:
    import networkx as nx

    cfg = ctx.cfg
    delta = int(obtener(cfg, "coord.delta_s", 300))
    bonus = float(obtener(cfg, "coord.bonus_texto", 1.0))
    min_tok = int(obtener(cfg, "coord.min_tok_texto", 6))
    alfa = float(obtener(cfg, "coord.alfa", 0.01))
    min_pub = int(obtener(cfg, "coord.min_pub", 2))
    min_size = int(obtener(cfg, "coord.min_size", 3))
    gamma = float(obtener(cfg, "coord.gamma", 1.0))
    rng = np.random.default_rng(int(obtener(cfg, "coord.semilla", 42)))

    com = ctx.db.leer("com")[["comentario_id", "cuenta_id", "publicacion_id", "ts", "h_esq", "h_texto",
                              "n_tok", "respuesta_a", "texto"]]
    ev = eventos(com)
    acc_codes, cuentas = pd.factorize(ev["cuenta_id"])
    post_codes, post_ids = pd.factorize(ev["publicacion_id"])
    ts = ev["ts"].to_numpy(np.int64)
    txt = np.where(ev["n_tok"].to_numpy() >= min_tok, ev["h_esq"].to_numpy(np.int64), 0)
    n_acc = len(cuentas)

    orden = np.lexsort((ts, post_codes))
    P, T, A, X = post_codes[orden], ts[orden], acc_codes[orden], txt[orden]
    k_obs, n_obs, s_obs, w_med = pares(P, T, A, X, delta, bonus, n_acc)

    # Nulo. "permuta" (defecto): cada lugar (publicación, momento) queda fijo y se baraja QUÉ CUENTA lo ocupa
    # entre los eventos de la misma fuente, período (coord.nulo_dias) y banda de demora (coord.nulo_demoras).
    # Conserva el tamaño de cada publicación, cuánto comenta cada cuenta y qué tan rápido (hiperactivos,
    # madrugadores) y rompe solo la elección conjunta. El texto viaja con la cuenta.
    # "reasigna": mueve cada evento a otra publicación de la misma fuente y período conservando la demora.
    reps = int(obtener(cfg, "coord.nulo_reps", 3))
    modo = str(obtener(cfg, "coord.nulo", "permuta"))
    tam = np.bincount(P, minlength=len(post_ids))
    ini_post = pd.Series(T).groupby(P).min().reindex(range(len(post_ids))).to_numpy()
    demora = T - ini_post[P]
    fuente = pd.Series("", index=post_ids)
    if ctx.db.existe("pub"):
        fuente = ctx.db.leer("pub").set_index("publicacion_id")["fuente_id"].reindex(post_ids).fillna("")
    fuente_cod = pd.factorize(pd.Series(fuente.to_numpy(), dtype=str))[0].astype(np.int64)
    dias = float(obtener(cfg, "coord.nulo_dias", 3))
    periodo = ((ini_post - np.nanmin(ini_post)) // max(1, int(dias * 86400))).astype(np.int64)
    estrato_post = fuente_cod * 100_000 + periodo
    if modo == "permuta":
        cortes = np.array(obtener(cfg, "coord.nulo_demoras", [600, 3600, 21600]), dtype=np.int64)
        clave = estrato_post[P] * 10 + np.searchsorted(cortes, demora, side="right")
        canon = np.lexsort((np.arange(len(P)), clave))
        n_estr = len(np.unique(clave))
    else:
        n_estr = len(np.unique(estrato_post))
    partes, nulos_pta = [], []
    for _ in range(reps):
        if modo == "permuta":
            azar = np.lexsort((rng.random(len(P)), clave))
            A0, X0 = np.empty_like(A), np.empty_like(X)
            A0[canon], X0[canon] = A[azar], X[azar]
            P0, T0 = P, T
        else:
            P0 = np.empty(len(P), dtype=np.int64)
            est_ev = estrato_post[P]
            for e in np.unique(est_ev):
                posts_e = np.flatnonzero(estrato_post == e)
                sel = est_ev == e
                P0[sel] = rng.choice(posts_e, size=int(sel.sum()), p=tam[posts_e] / tam[posts_e].sum())
            T0, A0, X0 = ini_post[P0] + demora, A, X
        nulos_pta.append((P0, T0, A0))
        nulo = pd.DataFrame({"p": P0, "t": T0, "a": A0, "x": X0}).drop_duplicates(["a", "p"])
        nulo = nulo.sort_values(["p", "t"], kind="stable")
        k0, n0, s0, _ = pares(nulo["p"].to_numpy(), nulo["t"].to_numpy(), nulo["a"].to_numpy(),
                              nulo["x"].to_numpy(), delta, bonus, n_acc)
        partes.append((k0, n0, s0))
    log.info("e08: nulo '%s' con %d estratos", modo, n_estr)
    kc = np.concatenate([x[0] for x in partes]) if partes else np.empty(0, np.int64)
    if len(kc):
        k_nul, inv = np.unique(kc, return_inverse=True)
        n_nul = np.bincount(inv, weights=np.concatenate([x[1] for x in partes])).astype(np.int64)
        s_nul = np.bincount(inv, weights=np.concatenate([x[2] for x in partes])) / reps
    else:
        k_nul, n_nul, s_nul = np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    # ε = peso típico de UNA co-ocurrencia: escala sin unidades que no depende del tamaño del set
    eps = w_med * float(obtener(cfg, "coord.eps_factor", 1.0))
    pos = np.searchsorted(k_nul, k_obs)
    pos = np.clip(pos, 0, max(len(k_nul) - 1, 0))
    esperado = np.where((len(k_nul) > 0) & (k_nul[pos] == k_obs), s_nul[pos], 0.0) if len(k_nul) else np.zeros(len(k_obs))
    lift = s_obs / (esperado + eps)

    # Respuestas repetidas entre las mismas dos cuentas (C2 de práctica "se responden entre sí")
    min_rp = int(obtener(cfg, "coord.min_resp_par", 2))
    peso_r = float(obtener(cfg, "coord.peso_respuesta", 2.0))
    padre = com[["comentario_id", "cuenta_id"]].rename(columns={"comentario_id": "respuesta_a", "cuenta_id": "c_p"})
    r = com[com["respuesta_a"] != ""][["cuenta_id", "respuesta_a"]].merge(padre, on="respuesta_a")
    r = r[r["cuenta_id"] != r["c_p"]]
    cod = pd.Index(cuentas)
    ru, rv = cod.get_indexer(r["cuenta_id"]), cod.get_indexer(r["c_p"])
    ok = (ru >= 0) & (rv >= 0)
    kr = np.empty(0, np.int64)
    if ok.any():
        kr, nr = np.unique(np.minimum(ru[ok], rv[ok]).astype(np.int64) * n_acc + np.maximum(ru[ok], rv[ok]),
                           return_counts=True)
        kr, nr = kr[nr >= min_rp], nr[nr >= min_rp]
        if len(kr):
            k_obs, n_obs, s_obs, esperado = sumar_evidencia(k_obs, n_obs, s_obs, esperado, kr, nr, peso_r * eps)
        log.info("e08: %d pares de cuentas con >= %d respuestas mutuas", len(kr), min_rp)

    # Hashtags poco usados compartidos (C1 de práctica: "hashtag propio")
    max_tag = max(min_size, int(float(obtener(cfg, "coord.hashtag_max_frac", 0.01)) * n_acc))
    kh, nh, n_tags = pares_hashtag(com, cod, n_acc, max_tag)
    if len(kh):
        k_obs, n_obs, s_obs, esperado = sumar_evidencia(k_obs, n_obs, s_obs, esperado, kh, nh,
                                                        float(obtener(cfg, "coord.peso_hashtag", 2.0)) * eps)
    log.info("e08: %d hashtags raros (<= %d cuentas) -> %d pares con evidencia de hashtag", n_tags, max_tag, len(kh))
    lift = s_obs / (esperado + eps)

    lift_min = float(obtener(cfg, "coord.lift_min", 4.0))
    fuerte = (n_obs >= min_pub) & (lift >= lift_min)
    theta = lift_min
    u, v = k_obs[fuerte] // n_acc, k_obs[fuerte] % n_acc
    log.info("e08: %d eventos, %d pares obs (%d con n_pub>=%d), nulo %d pares/rep, ε=%.3f -> %d aristas (lift>=%.1f)",
             len(ev), len(k_obs), int((n_obs >= min_pub).sum()), min_pub, len(k_nul) // max(reps, 1), eps,
             int(fuerte.sum()), lift_min)

    G = nx.Graph()
    G.add_weighted_edges_from(zip(u.tolist(), v.tolist(), lift[fuerte].tolist()))
    grupo = {}
    diag = []  # (grupo, cuentas, coh_obs, coh_nulo) por comunidad aceptada: lo usa el reporte
    if G.number_of_edges():
        # Poda por triángulos: una arista de campaña vive dentro de un bloque donde todos coinciden con todos;
        # una coincidencia casual casi no tiene vecinos en común.
        sop_min = float(obtener(cfg, "coord.soporte_min", 0.3))
        quitar = []
        for x, y in G.edges():
            m = min(G.degree(x), G.degree(y)) - 1
            comunes = len(set(G[x]) & set(G[y]))
            if m <= 0 or comunes / m < sop_min:
                quitar.append((x, y))
        H = G.copy()
        H.remove_edges_from(quitar)
        H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0])
        dens_min = float(obtener(cfg, "coord.densidad_min", 0.1))
        # Cohesión relativa al nulo: entra un grupo solo si sus miembros co-actúan (misma publicación,
        # |Δt| <= delta_s) más que esas mismas cuentas bajo las réplicas del nulo. Una campaña coordina
        # el QUÉ y el CUÁNDO; un grupo de genuinos hiperactivos coincide por volumen, no por elección.
        cohesion_min = float(obtener(cfg, "coord.cohesion_min", 0.0))
        # Actividad por cuenta: un grupo de hiperactivos coincide por estar en todos lados
        act = np.bincount(A, minlength=n_acc)
        q_act = float(obtener(cfg, "coord.max_actividad_q", 0.99))
        tope_act = np.quantile(act, q_act) if q_act < 1 else np.inf
        descartados_act = descartados_coh = 0
        resumen, resumen_coh, validas = [], [], []
        for comp in nx.connected_components(H):
            sub = H.subgraph(comp)
            # solo se parten componentes ralas (pueden ser dos operaciones pegadas); un bloque denso queda entero
            if len(comp) < 2 * min_size or nx.density(sub) >= float(obtener(cfg, "coord.densidad_partir", 0.5)):
                partes = [comp]
            else:
                partes = nx.community.louvain_communities(sub, weight="weight", resolution=gamma,
                                                          seed=int(obtener(cfg, "coord.semilla", 42)))
            for c in partes:
                if len(c) < min_size:
                    continue
                dens = nx.density(H.subgraph(c))
                med_act = float(np.median(act[list(c)]))
                resumen.append((len(c), round(dens, 2), round(med_act)))
                if med_act > tope_act:
                    descartados_act += 1
                elif dens >= dens_min:
                    co = cn = None
                    if cohesion_min > 0:
                        m = np.fromiter(c, dtype=np.int64)
                        co = cohesion(P, T, A, m, delta)
                        cn = float(np.mean([cohesion(p0, t0, a0, m, delta)
                                            for p0, t0, a0 in nulos_pta])) if nulos_pta else 0.0
                        resumen_coh.append((len(c), round(co, 2), round(cn, 2)))
                        if co - cn < cohesion_min:
                            descartados_coh += 1
                            continue
                    validas.append((c, co, cn))
        log.info("e08: poda por triángulos (soporte>=%.2f) %d -> %d aristas; comunidades (tamaño, densidad, "
                 "actividad mediana) %s; %d descartadas por actividad > p%g (%.0f publicaciones)", sop_min,
                 G.number_of_edges(), H.number_of_edges(), sorted(resumen, reverse=True)[:12], descartados_act,
                 100 * q_act, tope_act)
        if cohesion_min > 0:
            log.info("e08: cohesión por grupo candidato (n, obs, nulo) %s; %d descartadas por cohesión "
                     "(exceso < %.2f)", sorted(resumen_coh, reverse=True), descartados_coh, cohesion_min)
        fusion = str(obtener(cfg, "coord.fusion", "false")).lower()
        if fusion != "false" and len(validas) > 1:
            pasa = pares_fusion(ctx, validas, fusion, com, cuentas, P, T, A, nulos_pta,
                                delta, cohesion_min)
            comp = fusionar_grupos(validas, pasa)
            nuevas = []
            for cs in orden_grupos(comp, cuentas):
                originales = [vc for vc in validas if vc[0] <= cs]
                if len(originales) == 1:
                    nuevas.append(originales[0])
                else:
                    m = np.fromiter(cs, dtype=np.int64)
                    co = cohesion(P, T, A, m, delta) if cohesion_min > 0 else None
                    cn = (float(np.mean([cohesion(p0, t0, a0, m, delta)
                                         for p0, t0, a0 in nulos_pta]))
                          if cohesion_min > 0 and nulos_pta else None)
                    nuevas.append((cs, co, cn))
            validas = nuevas
            log.info("e08: fusión '%s': %d fusiones", fusion, len(pasa))
            orden = orden_grupos({i: vc[0] for i, vc in enumerate(validas)}, cuentas)
        else:
            orden = sorted(validas, key=lambda x: len(x[0]), reverse=True)
        for i, vc in enumerate(orden):
            c, co, cn = vc
            gid = f"g{i + 1:03d}"
            diag.append({"grupo": gid, "cuentas": len(c),
                         "coh_obs": round(co, 4) if co is not None else None,
                         "coh_nulo": round(cn, 4) if cn is not None else None})
            for nodo in c:
                grupo[cuentas[nodo]] = gid
    grupos = pd.DataFrame({"cuenta_id": list(grupo), "grupo": list(grupo.values())})
    pares_df = pd.DataFrame({"u": np.asarray(cuentas[u], dtype=object), "v": np.asarray(cuentas[v], dtype=object),
                             "n_pub": n_obs[fuerte].astype(np.int64), "score": s_obs[fuerte].astype(float),
                             "lift": lift[fuerte].astype(float)})
    # Fila centinela si está vacío: DuckDB no puede inferir tipos de columnas vacías (e09 la descarta)
    ctx.db.guardar("grupos", grupos if len(grupos) else pd.DataFrame({"cuenta_id": [""], "grupo": [""]}))
    diag_df = pd.DataFrame(diag, columns=["grupo", "cuentas", "coh_obs", "coh_nulo"])
    ctx.db.guardar("grupos_diag", diag_df if len(diag_df) else
                   pd.DataFrame({"grupo": [""], "cuentas": [0], "coh_obs": [0.0], "coh_nulo": [0.0]}))
    ctx.db.guardar("coord_pares", pares_df if len(pares_df) else
                   pd.DataFrame({"u": [""], "v": [""], "n_pub": [0], "score": [0.0], "lift": [0.0]}))
    sel = n_obs >= min_pub
    ctx.db.guardar("coord_pares_todos", pd.DataFrame({
        "u": np.asarray(cuentas[k_obs[sel] // n_acc], dtype=object), "v": np.asarray(cuentas[k_obs[sel] % n_acc], dtype=object),
        "n_pub": n_obs[sel].astype(np.int64), "score": s_obs[sel], "esperado": esperado[sel], "lift": lift[sel]})
        if sel.any() else pd.DataFrame({"u": [""], "v": [""], "n_pub": [0], "score": [0.0], "esperado": [0.0], "lift": [0.0]}))
    tam = grupos["grupo"].value_counts().tolist() if len(grupos) else []
    log.info("e08: %d cuentas coordinadas en %d grupos %s", len(grupos), len(tam), tam[:10])
