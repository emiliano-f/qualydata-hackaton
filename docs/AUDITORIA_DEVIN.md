# Auditoría independiente — Devin, 2026-11-12

Escrita **antes** de leer `docs/HANDOFF_DEVIN.md` y `docs/PLAN_DE_ACCION.md`. Fuentes: `qd/`, `herramientas/`,
`tests/`, `Makefile`, `config.yaml`, `evaluacion/puntuar.py`, `docs/DESAFIO.md`, `datos/practica` (con verdad), el
README y `reportes/practica.html` (captura en Firefox headless). Todo corrió en esta máquina (12 núcleos, 15 GB, sin
GPU, **sin `LLM_API_KEY`**: todas las corridas son sin IA). Scripts y artefactos de análisis: `/tmp/aud/` (no se suben).

## 0. Línea base reproducida

| Corrida | Narr /30 | Coord /25 | Total /55 | Tiempo | RSS máx |
|---|---|---|---|---|---|
| `make score` práctica (e5, caché de embeddings caliente) | 27,63 | 21,15 | **48,78** | 41 s | 1,8 GB |
| pseudo-final `p10` (10 días, 75 % pubs = **98.312** comentarios) | 27,89 | 16,32 | **44,21** | 23 s | 1,2 GB |
| `make humo` (juguete) | 23,77 | 20,54 | 44,31 | 7 s | 0,8 GB |
| `make test` / `make test-sucio` | ok | — | VALIDACIÓN OK | 4 s / 59 s | — / 3,0 GB |

Los números del README (48,78 y 44,21 sin IA) se reproducen exactamente. Desglose en práctica:
B-cubed 0,980 · ruido 0,985 · **postura 0,672** · origen 0,990 · coord P 0,730 / R 0,965 / F1 0,831 · grupos 0,880.

## 1. Riesgos de que `make final` falle o pierda puntos en silencio

Ordenados por puntos en juego × probabilidad.

**R1 · La orientación de la postura es una moneda por narrativa (hasta ~1,9 pts).** La partición en bandos es casi
perfecta (acierto por narrativa ≈ 1,00 o ≈ 0,00, nunca intermedio); lo que falla es *qué bando es "rechaza"*.
Práctica: 4/13 invertidas (B2, B3, H2, L3). p10: 2/13 (B2, H2), y **no son las mismas**: B3 y L3 se aciertan en p10.
Las cuatro tienen la afirmación formulada en negativo ("no quedan", "faltan médicos", "servicio pésimo"), así que el
voto de negación apunta al revés. Busqué en la verdad señales estructurales que generalicen (orden de aparición, primeros
20 comentarios, likes, respuestas recibidas, largo, antigüedad y seguidores por bando, tamaño de cada bando): **ninguna
separa de forma consistente** (`/tmp/aud/orientacion.py`). Además está acoplada a coordinación: con `coord.min_size=8`
la postura pasa de 0,672 a 0,745 sin tocar e06, solo porque cambia el voto `coord`. Sin LLM, el puntaje de postura del
final es básicamente azar entre ~0,5 y ~1,0.

**R2 · Sin clave, el caché commiteado de LLM no se usa: el jurado no reproduce la entrega.** `pipeline.py:88` pone
`sin_llm=True` si falta `LLM_API_KEY`, y e06 (`L242`) y e07 (`L55`) ni instancian `LLM`, así que `cache/llm/final/`
nunca se lee. `LLM.chat()` sí soporta modo solo-caché (lee el caché antes de mirar `degradado`), pero no se llega.
El README afirma lo contrario ("el jurado reproduce la entrega sin clave y sin costo"). Si el jurado re-corre sin
clave, obtiene otra postura y otros títulos que los de `entrega/`.

**R3 · Con clave, el LLM puede fallar en silencio (hipótesis con evidencia).** El `reportes/practica.html` commiteado
viene de una corrida con `modo_llm=completo`, 26 llamadas y 17.216 tokens, pero **todos los títulos son de TF-IDF**
("bombeo / agua / ver vegas"): `titular()` devolvió `None` siempre. Hipótesis: respuestas vacías o truncadas con
`max_tokens` 120/150 (los modelos con razonamiento consumen ese presupuesto antes de escribir), o JSON no parseable.
Lo mismo puede estar degradando `orientar()` a votos sin aviso (devuelve `None` → `llm_p+` vacío). Cómo verificarlo:
abrir `cache/llm/<set>/*.json` y buscar `"texto": ""` o JSON cortado; mirar `llm_p+` en `bandos_diag`. No pude
verificar si el modelo `gemini-3.8-flash` existe ni sus precios.

**R4 · La coordinación no generaliza a menos datos (−4,8 pts de práctica a p10).** Dos problemas que se suman:
- *Falsos positivos sistemáticos:* 7 grupos de 5–7 cuentas **genuinas** en práctica (41 cuentas) y 7 en p10 (36). Son
  cuentas viejas (mediana ~2.900 días) e hiperactivas (n_pub mediana 28 contra 6 del resto, 19–47 publicaciones), sin
  ninguna señal de campaña: coincidencias por azar entre millones de pares (multiple testing). El filtro
  `max_actividad_q=0,97` no las saca porque su actividad mediana queda por debajo del p97.
- *Fragmentación de campañas:* C2 sale partida en 12+9+7 (práctica) y 7+6 (p10); C3 en 6+5 en p10, con 12 cuentas
  perdidas. En p10 se pierden 38 de 113 cuentas de campaña. Los pedazos menores a `min_size=5` desaparecen.
El final tiene 10 días como p10 (aunque ~150k comentarios, más que los 98k de p10): es más probable que se parezca a
p10 que a práctica.

**R5 · `costo.json` puede contradecirse o subestimar el costo.** Evidencia en el reporte commiteado:
`notas="sin IA"` con `modo_llm=completo`, 26 llamadas y `usd_ia=0.0` con 17k tokens. Causas en el código:
(a) e02 sale temprano al reutilizar embeddings y nunca agrega su nota → si no hay otras notas queda "sin IA";
(b) los tokens que salen del caché no se cuentan → si se corre `make final` dos veces, la segunda declara 0 tokens y
0 USD; (c) `segundos` de una re-corrida caliente (~40 s) no es el costo real en frío (~8 min). El jurado evalúa
justamente esto.

**R6 · `make final` deja `pipeline.log` dentro de `entrega/`.** `configurar_log` escribe en `--salida`. Git lo ignora
(`*.log`) así que no se commitea, pero `entrega/` local tiene 5 archivos y `.gitignore` además habilita
`entrega/*.html` y `entrega/*.md`. Verificar con `git status entrega/` antes del commit final.

**R7 · Supuestos de práctica incrustados como hechos.** El ruido "nunca responde ni recibe respuestas" (e02b, e04),
"toda respuesta es de la narrativa del padre" (`heredar_todas`), "≤3 tokens acompaña, más largo invierte" (e06). Si el
generador del final es el mismo (lo más probable), suman mucho; si cambia, la pérdida es grande y silenciosa: el
validador solo avisa (rangos de `sanidad`). Mirar en el log del final `p_involucrado`, `% ruido` y nº de narrativas.

**R8 · Entorno.** `.python-version` fija 3.14 y uv resolvió pandas 3.0; con Python ≤3.10 (el enunciado dice 3.9+;
`pyproject` exige ≥3.10) `requirements.txt` instala pandas 2.3, un camino que nadie probó. e5 necesita el modelo de HF
(está cacheado acá). Correr el final en esta misma máquina con `uv.lock`.

**Riesgos bajos verificados:** lectura tolerante (BOM, líneas rotas, columnas faltantes, fechas inválidas, IDs
duplicados) cubierta por `test-sucio`; validador estricto en columnas, universo de IDs, posturas, fechas ISO `-03:00`,
etiquetas ⊂ narrativas; fallas de etapa terminan con exit 1 (no hay entrega parcial silenciosa salvo archivos viejos
de una corrida anterior en `entrega/`). El respaldo e5→TF-IDF no es un riesgo: ver O3.

## 2. Dónde hay más puntos (medido en práctica y p10, sin IA)

| # | Oportunidad | Práctica | p10 | Evidencia / riesgo de sobreajuste |
|---|---|---|---|---|
| O1 | Sacar FP de coordinación por **cohesión de grupo** | +2,6 (est.) | +2,2 (est.) | Fracción de eventos del grupo con otro miembro a ±Δ: campañas (y sus pedazos) **≥ 0,33** (C1 0,85, C2 0,33–0,45, C3 1,00); FP **≤ 0,22** en ambos sets (`/tmp/aud/cohesion.py`). Margen angosto: mejor relativo al nulo que un umbral fijo. |
| O2 | No partir componentes de campaña (`coord.gamma` 0,3 o `densidad_partir` 0) | **+0,64** | **+1,44 / +1,14** | Medido con `barrido`. gamma 0,3: 49,42 / 45,65. Junta C2. |
| O3 | Embeddings TF-IDF de caracteres en vez de e5 | **+0,75** | **+0,24** | B-cubed 0,996 vs 0,980 y 0,969 vs 0,962; origen 1,00 vs 0,99. e02 baja de ~8 min en frío a 16 s. Parte de la diferencia en postura es suerte de orientación. Riesgo: el set final podría parafrasear más que las plantillas del generador. |
| O4 | Orientación de postura correcta | techo +1,9 | techo +1,3 | Solo con señal semántica (LLM). No medible acá sin clave. Ajustar votos a práctica es sobreajuste. |
| O5 | Narrativas residuales | ≤ +0,3 | ≤ +0,6 | n001 absorbe ~1.400 comentarios ajenos; en p10, 622 de L2 caen en H1. |
| ✗ | `coord.min_size=8` | +2,44 | **−0,41** | Saca FP en práctica pero pierde pedazos de C2/C3 en p10: sobreajuste. Descartado. |

Ruido (0,985) y origen (0,99) ya están casi en el techo.

## 3. Cómo lo vería el jurado (criterios no automáticos)

**Costo, escala y almacenamiento (20).** Fortalezas: `costo.json` automático con desglose por etapa; deduplicación
(29 % de textos únicos); IA proporcional a narrativas, no a comentarios (~40 llamadas); 1,8 GB de RSS; DuckDB embebido
bien justificado. Debilidades: las contradicciones de R5 son lo primero que se ve en el reporte; el Parquet por día se
exporta pero ninguna etapa lo lee (el relato de almacenamiento queda decorativo); el plan de escala no trae números
medidos (comentarios/s por etapa) y no reconoce que e08 es cuadrático en el tamaño de ventana de una publicación viral.
Si TF-IDF rinde igual o mejor, justificar e5 frente a un plan B 30 veces más barato es difícil.

**Robustez (15).** Un comando (`make final`, después de `make instalar`), datos sucios ok, semillas fijas y números del
README reproducidos al centésimo. El punto débil es R2: con IA, la reproducción sin clave no funciona aunque el README
diga que sí. Los tests (11, 4 s) no cubren el camino LLM/caché.

**Claridad (10)** (captura de `reportes/practica.html` commiteado). Se entiende la estructura (qué circula → quién
empuja → costo) y la red de grupos. Pero en un minuto no se entiende *qué* circula: los títulos son bolsas de palabras
("gracias / puestos / 300", "totalmente / 100 / exacto"); aparecen narrativas basura (n013, n018 con 3 comentarios);
"Quién la empuja" muestra un falso positivo (g007, 0,4 % de los comentarios) en 7 narrativas, lo que tapa las dos
señales reales (g001 → n004 "trabajo", 87 % apoya; g002 → la emergente n014, 100 % apoya); las líneas de origen de las
emergentes casi no se ven. La tabla de costo muestra `notas: sin IA` junto a `modo_llm: completo`. Nota: correr
`make score` con plotly instalado regenera el archivo con plotly embebido (170 KB → 5 MB).

## 4. El README leído desde afuera

**Se entiende:** cómo correrlo, la tabla de etapas, por qué DuckDB + Parquet, que la IA es opcional y qué librerías
existían antes. La tabla de resultados es concreta.

**No se entiende o no cierra:**
- Promete reproducción sin clave (falso hoy, R2) y "4 archivos" en `entrega/` (quedan 5 en disco, R6).
- Reporta B-cubed, ruido, origen y F1 de coordinación, pero **no la postura (0,67)**, que es la parte débil. El jurado
  va a ver el puntaje real; mejor declararlo con su causa.
- No dice dónde queda el reporte del final (`reportes/final.html`) ni cómo leerlo, ni qué narrativas y operaciones se
  encontraron en práctica (el "qué circula" en una línea).
- La tabla de etapas sigue el orden de ejecución (e08 antes de e06) sin decir por qué; falta e05 (hueco en la numeración).
- "Con IA 49,37" no es verificable sin la clave ni el caché de práctica.
- Escala: describe arquitectura pero sin cifras propias (comentarios/s, memoria) y sin cuello de botella identificado.
- No hay sección de limitaciones (orientación de postura, FP de coordinación con menos datos).

## 5. Prioridades sugeridas antes del congelamiento (22 h)

1. **O2** (`coord.gamma: 0.3`): cambio de configuración, medido en práctica y p10, mejora ambos.
2. **O1**: filtro de cohesión de grupo relativo al nulo. Solo entra si mejora práctica y no empeora p10.
3. **R2 + R3 + R5** (reproducción solo-caché, LLM vacío visible, costo honesto): no mueven `make score`, así que según
   la regla acordada necesitan una excepción explícita. Son los que más protegen robustez y costo frente al jurado.
4. **O3** (TF-IDF): decisión de diseño. La evidencia es favorable en los dos sets, pero cambia el motor principal un día
   antes; requiere el visto bueno del equipo.
5. README: postura real, limitaciones, reproducción, dónde está el reporte. Reporte: "Quién la empuja" solo por encima
   de lo esperable y ocultar narrativas diminutas.
