# Traspaso del proyecto QualyData a Devin

> Estado al jueves 12/11/2026, 13:20 (hora argentina). El set final se publica el **viernes 13 a la mañana**;
> lo que cuenta es lo que esté en el repo público al cierre (la organización clona y anota el commit).
> Este documento cubre lo que **no** está en el código: estado, decisiones y su porqué, hallazgos de los datos,
> pendientes priorizados y el procedimiento del viernes. La arquitectura está en `README.md`; el plan original
> (con decisiones superadas) en `docs/PLAN_DE_ACCION.md`.

---

## 1. Estado actual

| Corrida (puntaje automático oficial, `evaluacion/puntuar.py`) | Narrativas /30 | Coordinación /25 | Total /55 |
|---|---|---|---|
| Línea base del repo del desafío | — | — | ~17 |
| Práctica, sin LLM (`--sin-llm`) | 27,63 | 21,15 | **48,78** |
| Práctica, con LLM (`gemini-3.5-flash-lite`) | 28,22 | 21,15 | **49,37** |
| Pseudo-final sin LLM (10 días, 75% de publicaciones) | 27,89 | 16,32 | 44,21 |

Detalle práctica sin LLM: B-cubed 0,980 (P 0,983 / R 0,978) · ruido F1 0,985 · postura macro-F1 0,67 ·
origen 0,99 · coordinación F1 0,83 (P 0,73 / R 0,965) · grupos B-cubed 0,88.

- Repo limpio y pusheado; dependencias consolidadas con uv (`pyproject.toml` + `uv.lock`; `requirements.txt`
  generado con `uv export`). **Ensayo en clon limpio completo (12/11 13:44)**: instalación, `make test`,
  `make humo`, `make test-sucio`, `make score` en frío y `make final` sobre el pseudo-final →
  `VALIDACIÓN OK`; `git status` mostró solo `entrega/`, `reportes/final.html` y `cache/llm/` (lo esperado).
- **Hallazgo del ensayo**: el pipeline tardó ~20 s y el LLM 19 min (`e06_postura` 1.146 s) para terminar en modo
  degradado con 1 sola llamada exitosa (Gemini 503 + timeouts; con `reintentos: 4` y `timeout_s: 60` cada llamada
  puede tardar ~5 min antes de contar como falla). **Acción inmediata: `llm.activo: false` en `config.yaml`**
  (ver §5.2).
- Entorno del autor: Fedora, Python 3.14 (uv), CPU sin GPU, torch CPU. Embeddings e5 en frío: ~8,5 min para
  58k textos únicos (114 textos/s); corrida completa en caliente ~35 s.

## 2. Reglas del evaluador que condicionan todo (leídas de `evaluacion/puntuar.py`)

1. **Postura siempre completa**: el macro-F1 se mide sobre todos los comentarios con narrativa real, sin mirar
   la narrativa predicha. Postura vacía = falso negativo. Por eso `e09_exporta` nunca deja postura vacía.
2. **Ruido asimétrico**: B-cubed solo mira comentarios con narrativa real; un comentario marcado ruido cuenta
   como singleton (pierde recall). Meter ruido en un clúster no cuesta B-cubed → marcar ruido con confianza alta.
3. **Origen**: a cada narrativa emergente real se le asigna el clúster predicho con más comentarios en común;
   si una emergente queda absorbida por un clúster viejo, saca 0.
4. **Fechas** `YYYY-MM-DDTHH:MM:SS-03:00` sin fracción: fecha sin zona → `TypeError` y el script se cae.
5. **`"nan"` no cuenta como vacío** (sí `""`, `ninguna`, `none`, `null`): en `cuentas.csv` marcaría a todos como
   coordinados. El validador (`qd/validar.py`) lo bloquea.
6. **Coordinación**: cualquier grupo no vacío = marcada. Los falsos positivos no afectan el B-cubed de grupos;
   fusionar dos campañas lo destruye.

## 3. Hallazgos de los datos de práctica (estructurales, se espera que valgan en el final)

- 13 narrativas (3 emergentes: L5 día 6, B4 día 9,6, H3 día 4,7), 18,9% de ruido, 3 temas (L litio, B, H salud).
- `fecha_origen` real = primer comentario de la narrativa.
- **Las respuestas (14%) son SIEMPRE de la narrativa del padre y nunca son ruido; nadie le responde al ruido.**
  Base del ruido estructural (`e02b`) y de la fusión de narrativas por respuestas (`e04`).
- **Postura**: respuestas cortas (≤3 tokens, "tal cual", "exacto"… con typos) acompañan al padre (100%);
  respuestas largas lo contradicen (80% de todas las respuestas son de postura opuesta). Los rechazos son
  contra-afirmaciones, no negaciones (el léxico de negación cubre solo 15,6%). Base de la bipartición de bandos.
- **Campañas** (115 cuentas = 0,72%, todas empujan "apoya"):
  - C1: 60 cuentas nuevas, texto casi idéntico con hashtag propio (`#ellitioesfuturo`), ráfagas diurnas de
    12 min, rota subconjuntos (un par coincide ~1 vez) → la detecta la evidencia de hashtag.
  - C2: 30 cuentas antiguas, textos variados, mismas publicaciones, se responden entre sí, gaps ~250 s
    → la más difícil (hoy se detectan ~18 de 30).
  - C3: 25 cuentas nuevas, ráfagas nocturnas (01–05 h), texto casi idéntico, originan la emergente H3.
- **Trampas**: picos virales genuinos (~450 comentarios en 10 min), plantillas de ruido usadas por 500–970
  cuentas, y genuinos hiperactivos (80–160 comentarios) que coinciden por estar en todo.

## 4. Decisiones calibradas (valores en `config.yaml`) y por qué

| Parámetro | Valor | Evidencia |
|---|---|---|
| `micro.k_factor` | 2 | barrido: k=1 → 32,4; k=2 → 36,2 (B-cubed 0,98) |
| `narr.gamma_resp` | 2 | Louvain sobre grafo de respuestas; más alto fragmenta |
| `coord.nulo` | `permuta` | +5,2 pts vs `reasigna`. Se baraja qué cuenta ocupa cada lugar (publicación, momento) dentro de fuente × período × banda de demora |
| `coord.delta_s` | 450 | 300 → 47,2; 450 → 48,8; 600 → 48,4 |
| `coord.max_actividad_q` | 0,97 | descarta grupos de hiperactivos (+2,6 pts vs 0,99) |
| `coord.min_size` | 5 | cliques chicas aparecen por azar |
| `postura.voto_*` | ref 3, neg 1, coord 2 (relativo ≥30%), temprano 0 | ref (refutación explícita) nunca falló; temprano falló en B4 |
| `llm.*` | Gemini endpoint OpenAI-compatible | ver §5.2 |

Descartado con evidencia (no volver a probar): UMAP+HDBSCAN; ruido por micro-clúster (las plantillas son pocos
textos únicos y no tienen centroide propio); nulo "permutar cuentas dentro de cada publicación" (conserva la
co-membresía); nulo "concurrente" (una ráfaga de campaña domina su franja como un viral → anula la detección);
LLM local Qwen 1,5B (no movió la postura, 6 min); prompt de orientación sin ejemplos (sesgo de posición).

## 5. Pendientes, en orden de prioridad

### 5.1 Ensayo general — HECHO (ver §1). Pendientes que dejó:
- Apagar el LLM por defecto (`llm.activo: false`) o hacerlo cortar rápido (`timeout_s: 20`, `reintentos: 1`,
  `max_fallas: 2`). Con el LLM apagado `costo.json` reporta el tiempo real del pipeline.
- `entrega/pipeline.log`: el log se escribe dentro de la carpeta de salida. No se commitea (`*.log` está en
  `.gitignore`), pero conviene moverlo a `work/<set>/pipeline.log` (`configurar_log` en `qd/pipeline.py`)
  para que `entrega/` tenga solo los 4 archivos.
- Volver a correr el ensayo de `make final` después de esos cambios (debería tardar < 1 min con embeddings en
  caché) y borrar el clon (`rm -rf /tmp/ensayo`). No pushear desde el ensayo.

### 5.2 Orientación de bandos por LLM (≈ +2 pts si funciona; opcional)
- La bipartición en bandos es casi perfecta (cada narrativa acierta 0,99 o 0,00): el único error es **cuál bando
  se llama "apoya"**. Sin LLM quedan 4 narrativas invertidas (B1, B2, H2, L3 en la última corrida).
- `qd/llm.py`: cliente stdlib a `/chat/completions`, caché en `cache/llm/<set>/`, reintentos, modo degradado.
  Prompt v2 con 3 ejemplos resueltos de práctica (`qd/fewshot_postura.json`, generado por
  `herramientas/fewshot.py`), pide `NARRATIVA:` y `RECHAZA: A|B`, en las dos posiciones; si se contradicen
  (`llm_p+ = 0.5`) decide por votos.
- Resultados: flash-lite → postura 0,771 (49,37); flash 3.6 → 0,672 (peor). Gemini con 503 y timeouts
  frecuentes (corridas de 8 min). **Ojo con la fuga**: los 3 ejemplos son narrativas de práctica; medir el
  acierto en las otras 10 (`python herramientas/diagnostico.py --datos datos/practica`, tabla de orientación).
- Ideas no implementadas: varias consultas con distintos subconjuntos de ejemplos y voto por mayoría; el LLM
  como voto con peso en vez de decisión; llamadas en paralelo; probar otro proveedor (OpenAI/Anthropic tienen
  endpoint compatible: solo cambian `llm.base_url` y `llm.modelo`).
- Sin LLM la entrega sale igual (≈ −0,6 pts en práctica). No arriesgar el viernes por esto.
- **Estado al 12/11 13:40**: el ensayo de `make final` con LLM volvió a sufrir "high demand" (503) de Gemini.
  Recomendación para el viernes: o bien dejar `llm.activo: false` por defecto (entrega determinista, ~48,8 en
  práctica), o bien que el LLM degrade rápido: `llm.timeout_s: 20`, `llm.reintentos: 2`, `llm.max_fallas: 2`
  (con eso, si la API no responde, pasa al respaldo en ~1 min en vez de ~8). Decidirlo hoy, no el viernes.

### 5.3 Coordinación en el set final
En el pseudo-final cae a 16,3 (F1 0,67), esperable con menos repetición; el final real es más denso
(~150k en 10 días). C2 sigue parcialmente sin detectar. Diagnóstico: `python herramientas/perfil_coord.py
--datos datos/practica` (por campaña: pares que repiten por ventana, hashtags, fuentes, horas; por grupo predicho:
fuente dominante, antigüedad, actividad). Cualquier cambio debe mejorar práctica **y** no empeorar el pseudo-final.

### 5.4 Demo y pitch (10 pts de jurado + requisito excluyente)
- **Abrir `reportes/practica.html` en el navegador y revisarlo** (nunca se vio renderizado): área apilada por
  narrativa con orígenes, red de operaciones, costos. Usa plotly.js embebido si `plotly` está instalado.
- Grabar el video de demo hoy con datos de práctica.

### 5.5 Detalles del repo
- `LICENSE` MIT (las bases lo proponen).
- README, "Cómo correrlo": aclarar que `make final` requiere `datos/final/` (se baja el viernes) y que
  `make score` requiere `datos/practica/` del repo del desafío (no se versiona: `.gitignore` excluye `*.csv.gz`).

## 6. Criterios del jurado (45 de los 100 puntos): qué está cubierto y qué falta

| Criterio | Pts | Qué evalúan |
|---|---|---|
| Costo, escala y almacenamiento | 20 | `costo.json` y tiempo de proceso; dónde y cómo guardan los datos, y por qué esa base; cómo escalaría a millones de comentarios por día |
| Robustez del pipeline | 15 | corre de punta a punta con un comando, maneja datos sucios y es reproducible |
| Claridad y visualización | 10 | demo y pitch: ¿se entiende en un minuto qué circula, desde cuándo y quién lo empuja? |

### 6.1 Costo, escala y almacenamiento (20)
Cubierto:
- `entrega/costo.json` se genera solo (`qd/costo.py`): las 8 claves pedidas + extras (`desglose_segundos` por
  etapa, `llamadas_llm`, `aciertos_cache_llm`, `modo_llm`, `modelo_llm`). Precios del LLM en `precios.yaml`.
- Almacenamiento: DuckDB embebido (`work/<set>/qd.duckdb`) + Parquet particionado por día; justificación y
  plan de escala a millones/día en `README.md` (secciones "Dónde se guardan los datos" y "Cómo escala").
- Costo de IA proporcional a la cantidad de **narrativas**, no de comentarios (~40 llamadas por set).
Revisar / falta:
- Que `costo.json` del final refleje la corrida real (borrar `cache/llm/final` antes; la caché marca 0 tokens).
- Cifras de tiempo del README con lo medido en el ensayo limpio (en frío ≈10 min dominado por embeddings e5 en
  CPU; en caliente ≈35 s). Si el final tarda distinto, actualizarlas.
- En el pitch: una cifra concreta de USD por 1.000 comentarios y el argumento de que embeddings se calculan una
  vez por texto único (deduplicación ~3×) y la asignación en línea es un producto punto contra centroides.
- Opcional: diagrama simple de la arquitectura (ingesta → DuckDB/Parquet → etapas → entrega) para el README o
  el video.

### 6.2 Robustez del pipeline (15)
Cubierto:
- Un comando: `make final` (incluye validación; sale con código ≠ 0 si algo haría perder puntos en silencio).
  `make instalar` (uv sync / pip).
- Datos sucios: `qd/entrada.py` tolera líneas rotas, BOM, columnas faltantes, fechas inválidas, IDs duplicados,
  `respuesta_a` a comentarios inexistentes; `make test-sucio` lo demuestra con `herramientas/ensuciar.py`.
- Reproducible: semillas fijas, `uv.lock`, caché del LLM de la corrida final commiteada (el jurado re-corre sin
  clave), respaldo determinista si no hay LLM, plan B de embeddings TF-IDF si no hay modelo.
- Ensayo en clon limpio: pasos 1–4 OK.
Revisar / falta:
- Cerrar el paso 5 del ensayo (§5.1) **con el LLM resuelto** (§5.2): hoy la alta demanda alarga la corrida.
- README: aclarar que `make final` necesita `datos/final/` y `make score` necesita `datos/practica/`.
- Mostrar en el video `make test-sucio` corriendo (10 segundos) como evidencia de robustez.

### 6.3 Claridad y visualización (10)
Cubierto:
- `reportes/<set>.html` se genera al final de cada corrida (`qd/etapas/e10_reporte.py`): indicadores, área
  apilada de narrativas cada 6 h con el origen de las emergentes, tabla de narrativas (título, volumen,
  % rechazo, origen, qué grupo la empuja), red de operaciones coordinadas con tabla por grupo, costos por etapa.
  Paleta categórica validada, modo oscuro, tablas como vista accesible.
Revisar / falta:
- **Nunca se vio renderizado**: abrir `reportes/practica.html` y revisar superposición de etiquetas, leyenda,
  tamaño de la red y legibilidad en pantalla de proyector.
- Títulos legibles: sin LLM los títulos son términos TF-IDF ("obra / plata / hospital"); con LLM son frases.
  Si el final corre sin LLM, considerar editar el texto del pitch en vez de mostrar esos títulos tal cual.
- Guion de 1 minuto: (1) qué circula → narrativas y su tamaño; (2) desde cuándo → emergentes y su origen;
  (3) quién lo empuja → grupos coordinados y qué narrativa empujan; (4) costo y tiempo. Grabar el video hoy con
  práctica y, si hay tiempo el viernes, cambiar las capturas por las del final.

## 7. Procedimiento del viernes (en el repo real, no en el ensayo)

```bash
# 1. Descargar el set final a datos/final/ y verificar los 4 archivos
ls datos/final/                                # fuentes.csv, publicaciones.csv.gz, comentarios.csv.gz, cuentas.csv.gz
rm -rf cache/llm/final entrega/*               # costo real y caché limpia
export LLM_API_KEY=...                         # opcional
time make final                                # ~10 min; termina en "VALIDACIÓN OK"
# Si el LLM se satura: Ctrl-C y  make final FLAGS="--sin-llm"
head entrega/narrativas.csv; cat entrega/costo.json
git add entrega/ cache/llm/final/ reportes/final.html
git commit -m "Entrega final" && git push
```
Después: verificar en GitHub que `entrega/` se ve en el repo público; cargar link + video en Hacki; dejar
≥ 30 min de margen. Los avisos de sanidad del validador (rangos 0,5×–2× de práctica) son solo avisos.

## 8. Qué NO hacer

- No poner la clave en ningún archivo (solo `LLM_API_KEY`); antes de pushear: `git grep -nE "AIza|sk-"`.
- No etiquetar a mano ni ajustar parámetros mirando el set final (prohibido por las bases; el jurado puede
  re-correr). Calibrar solo con práctica y el pseudo-final.
- No agregar archivos a `entrega/` además de los 4 (el reporte va a `reportes/`).
- No tocar `evaluacion/puntuar.py`.
- No cambiar el orden de etapas sin entenderlo: `e08_coord` corre **antes** de `e06_postura` porque la
  orientación de bandos usa las cuentas coordinadas.

## 9. Mapa rápido para trabajar

- Etapas (`qd/pipeline.py › ETAPAS`): e01 ingesta → e02 embed → e02b ruido → e03 micro → e04 narrativas →
  e08 coord → e06 postura → e07 origen → e09 exporta → e10 reporte. Cada una lee y escribe tablas en
  `work/<set>/qd.duckdb`; `--desde eNN` saltea las anteriores si sus tablas existen.
- `--set clave=valor` pisa `config.yaml`. `barrido.py` corre siempre sin LLM y registra en `experimentos.csv`.
- Loop: `make score` → `python herramientas/diagnostico.py --datos datos/practica` →
  `make barrido P=<param> V=<valores> ETAPA=<eNN>` → confirmar en `make score SET=p10 DATOS=work/datos/p10 SALIDA=salida/p10`.
- Pseudo-final: `python herramientas/pseudo_final.py --origen datos/practica --destino work/datos/p10`.
- Tests: `make test` (unitarios), `make humo` (juguete que reproduce la estructura de práctica),
  `make test-sucio` (datos rotos).
- Caché de embeddings: `cache/emb/<modelo>/` (claves = hash del texto normalizado). La caché del LLM depende
  del modelo y del prompt: cambiar cualquiera de los dos vuelve a llamar a la API.
