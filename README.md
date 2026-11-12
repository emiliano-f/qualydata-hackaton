# QualyData · narrativas y coordinación a escala — solución del equipo

Pipeline de un solo comando que descubre narrativas, asigna cada comentario a su narrativa o al ruido,
estima su postura y su origen, y detecta operaciones coordinadas. Corre en una notebook sin GPU,
**sin datos de entrenamiento del set final** y con IA solo en ~40 llamadas puntuales (opcional).

## Cómo correrlo

Requisitos: Python 3.10+ (el repo fija 3.14 en `.python-version`). Recomendado: [uv](https://docs.astral.sh/uv/).

```bash
make instalar                 # uv sync (usa uv.lock); sin uv: pip install -r requirements.txt
make final                    # datos/final -> entrega/ (4 archivos) + validación
make score                    # set de práctica + puntaje oficial
make humo && make test        # datos de juguete y tests (no necesitan el set real)
make test-sucio               # datos rotos a propósito: el pipeline no se cae
```

- **IA opcional y apagada por defecto** (`llm.activo: false`). Para prenderla: `llm.activo: true`, exportá
  `LLM_API_KEY` y completá `llm.base_url` / `llm.modelo` (cualquier API compatible con OpenAI). Corte rápido si la
  API falla (timeout 20 s, 2 reintentos, 2 fallas seguidas → respaldo determinista); las respuestas truncadas o vacías
  se descartan con un WARNING y no se guardan en caché.
- **Reproducible:** semillas fijas en todo. Con la IA prendida, las respuestas quedan en `cache/llm/<set>/`
  (se commitea `cache/llm/final/`). **Sin clave**, el pipeline lee ese caché y no hace llamadas de red. Lo probamos de
  punta a punta con un servidor falso: los 3 CSV salen byte-idénticos y `costo.json` declara los mismos tokens y USD,
  con `modo_llm: desde_cache`. `modo_llm` vale `completo`, `desde_cache`, `degradado` (falló la API o faltaba caché) o
  `apagado`.
- **Calibración:** `make barrido P=<param> V=<v1,v2,...> ETAPA=<eNN>`; `python herramientas/diagnostico.py`.
- **Reporte para la demo:** `reportes/<set>.html`, que se genera al final de cada corrida (`make final` escribe
  `reportes/final.html`). Se abre en el navegador y tiene tres secciones: qué circula y desde cuándo (narrativas en el
  tiempo, con el origen de las emergentes), quién lo empuja (red de cuentas coordinadas y a qué narrativa empuja cada
  grupo) y costo y escala. El log de cada corrida queda en `work/<set>/pipeline.log`, fuera de `entrega/`.
- **Dependencias:** `pyproject.toml` + `uv.lock` son la fuente única; `requirements.txt` se genera con
  `uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt`.

## Cómo funciona

| Etapa | Qué hace | Por qué funciona en un set nuevo |
|---|---|---|
| `e01_ingesta` | CSV → DuckDB (+ Parquet por día); fechas a epoch; textos normalizados y deduplicados | lectura tolerante: líneas rotas, fechas inválidas, IDs duplicados, columnas faltantes |
| `e02_embed` | `multilingual-e5-small` sobre **textos únicos** (29% del total), con caché | agrupa por significado, no por palabras |
| `e02b_ruido` | ruido por texto: el ruido **nunca responde ni recibe respuestas**; lo frecuente sin respuestas es ruido, lo raro se decide por vecinos | hecho estructural, independiente del tema |
| `e03_micro` | KMeans sobre-particionado (≈240) ajustado con muestra estratificada **por día** | las narrativas que nacen tarde tienen peso propio |
| `e04_narrativas` | aglomerativo con umbral relativo + **Louvain sobre el grafo de respuestas** (una respuesta siempre es de la narrativa de su padre) | une redacciones distintas y bandos opuestos de una misma narrativa |
| `e08_coord` | co-acción repetida (misma publicación, ±Δ) ponderada contra virales, vs **modelo nulo por permutación** (fuente × período × banda de demora); + respuestas mutuas + hashtags raros; poda por triángulos; Louvain; **cohesión de grupo vs el nulo** (un grupo entra solo si sus cuentas co-actúan ≥ 0,2 más que esas mismas cuentas en el nulo) | compara contra lo esperable: virales, hiperactivos y seguidores de una fuente no son campañas |
| `e06_postura` | respuestas cortas acompañan, largas contradicen → **bipartición del grafo de respuestas** en bandos; orientación por LLM (2 llamadas por narrativa) o por votos (refutación explícita, negación, coordinación) | la postura sale de la estructura de la conversación, no de palabras clave |
| `e07_origen` | k-ésimo comentario más temprano entre los centrales, con confirmación de ráfaga; títulos por LLM | rango (no percentil): no se corre con el tamaño |
| `e09_exporta` + validador | 4 archivos de `entrega/`; falla si algo haría perder puntos en silencio | postura nunca vacía, sin `nan`, fechas ISO `-03:00` |

La tabla sigue el orden de ejecución: `e08` corre antes que `e06` porque las cuentas coordinadas son uno de los votos
que orientan la postura. `e10_reporte` escribe el HTML. No hay `e05`.

## Dónde se guardan los datos y por qué DuckDB + Parquet

- **DuckDB** (`work/<set>/qd.duckdb`) para las tablas intermedias: motor columnar embebido, cero operación, SQL
  completo, rápido para agregaciones y self-joins sobre cientos de miles de filas en una notebook.
- **Parquet particionado por día** (`work/<set>/parquet/comentarios/dia=N/`): capa de archivo y de análisis. Ninguna
  etapa del pipeline lo lee (trabajan sobre DuckDB). Está para que cualquier herramienta (DuckDB, Spark, ClickHouse o
  BigQuery) consulte los comentarios ya normalizados sin conversión, y es el formato de la ingesta por partición del
  plan de escala. Elegimos no levantar un servidor (PostgreSQL/Elastic) porque para 150–200k comentarios agrega
  operación sin agregar capacidad.
- Embeddings en `cache/emb/` (claves = hash del texto normalizado): se calculan una vez por texto único y se
  reutilizan entre corridas y entre sets.

## Cómo escala a millones de comentarios por día

1. **Ingesta:** cola (Kafka/PubSub) → tabla particionada por día en ClickHouse o BigQuery (las mismas consultas).
2. **Embeddings:** solo textos nuevos (la deduplicación ya baja el volumen ~3×); micro-lotes en GPU (~3.000 textos/s).
3. **Narrativas en línea:** cada comentario nuevo se asigna al centroide más cercano (un producto punto); una vez por
   día se re-agrupa solo lo que quedó lejos de todo, que es donde aparecen las narrativas nuevas.
4. **Coordinación incremental:** la co-acción es un join por `publicacion_id` con rango de tiempo — se calcula por
   ventana deslizante de 24–48 h y se acumulan los pares; el nulo se recalcula por partición.
5. **IA:** su costo depende de la cantidad de **narrativas**, no de comentarios: no crece con el volumen.

**Medido en esta notebook** (12 núcleos, 15 GB, sin GPU, sin IA):

| Set | Comentarios | Tiempo (caché de embeddings caliente) | RSS máx |
|---|---|---|---|
| Práctica | 200.000 (14 días) | 36–41 s | 1,8 GB |
| Pseudo-final | 98.312 (10 días) | 20–23 s | 1,2 GB |
| Datos sucios (`make test-sucio`) | — | 59 s | 3,0 GB |

En frío, `e02_embed` (e5 en CPU sobre ~58k textos únicos) tarda ≈8 min y domina todo el costo. El resto procesa unos
5.000 comentarios/s. El próximo cuello de botella es `e08`: los pares de co-acción crecen con el cuadrado de los
comentarios que caen en la misma ventana de una publicación viral. Por eso, en producción va por ventanas deslizantes
(punto 4).

**e5 vs TF-IDF de caracteres.** El respaldo TF-IDF (3–5 gramas + SVD 256) calcula los embeddings en 16 s y dio un puntaje
equivalente en práctica (B-cubed 0,996 contra 0,980) y en el pseudo-final (0,969 contra 0,962). Igual mantenemos
e5 como motor principal: los datos de práctica salen de plantillas con variaciones ortográficas, que es justo lo que TF-IDF
captura. Un set con más paráfrasis (mismo significado con otras palabras) lo agrupa e5 y no TF-IDF. TF-IDF queda como
plan B automático si el modelo no carga (`--embed tfidf`).

## Costo (ver `entrega/costo.json`)

`costo.json` se arma solo en cada corrida: segundos por etapa, hardware, tokens y USD de IA. Los tokens incluyen el
razonamiento oculto que factura el proveedor, y las respuestas que salen del caché cuentan lo que costó producirlas,
así una re-corrida no declara costo cero. Sin IA, el costo es el tiempo de CPU de la tabla de arriba. Con IA son unas
40 llamadas y ~22k tokens, y el `usd_por_mil_comentarios` queda en milésimas de centavo.

## Resultados en el set de práctica (puntaje automático oficial)

| Corrida | Narrativas /30 | Coordinación /25 | Total /55 |
|---|---|---|---|
| Línea base del repo | — | — | ~17 |
| Nuestra solución, sin IA | 27,49 | 24,43 | **51,92** |
| Pseudo-final sin IA (primeros 10 días, 75% de las publicaciones) | 27,53 | 20,04 | **47,57** |
| Versión anterior, sin IA (antes del filtro de cohesión y `gamma` 0,3) | 27,63 | 21,15 | 48,78 |
| Versión anterior, con IA | 28,22 | 21,15 | 49,37 |

Práctica: B-cubed de narrativas 0,98 · ruido F1 0,985 · **postura macro-F1 0,65** · origen de emergentes 0,99 ·
coordinación P 1,00 / R 0,965 / F1 0,98 · grupos 0,965. Pseudo-final: postura 0,70, coordinación F1 0,83, grupos 0,74.
El pseudo-final (`herramientas/pseudo_final.py`) mide generalización con menos datos que el set final real.

La fila "con IA" es de la configuración anterior y no se volvió a medir. Además, las corridas con `gemini-3.8-flash`
salieron truncadas: el razonamiento oculto se comía el `max_tokens` de 120–150. Eso ya está corregido (1000 tokens,
`reasoning_effort: low`, y las respuestas truncadas se descartan), pero la mejora con IA no está medida en esta versión.

## Limitaciones conocidas

- **Postura (0,65 en práctica).** Los bandos de cada narrativa salen casi perfectos: dentro de una narrativa, los
  comentarios quedan casi todos bien separados o casi todos al revés. Lo que falla es decidir qué bando "rechaza".
  Sin IA, esa decisión sale de votos: refutación explícita, negación y cuentas coordinadas. Cuando la afirmación misma
  está formulada en negativo ("faltan médicos", "no quedan puestos"), el voto de negación apunta al revés: en la versión
  anterior se invertían 4 de las 13 narrativas de práctica. Como cada narrativa es todo o nada, el puntaje de postura del set final tiene mucha
  varianza. Es el lugar donde más puede aportar la IA.
- **Coordinación con menos datos.** En el pseudo-final (10 días) se pierde un cuarto de las cuentas de campaña
  (R 0,74): las campañas chicas co-actúan pocas veces y no llegan a la evidencia mínima. El filtro de cohesión saca
  los grupos falsos de cuentas genuinas hiperactivas (7 en práctica y 7 en el pseudo-final), pero no recupera las
  campañas que se pierden.
- **Supuestos de práctica.** "El ruido no responde ni recibe respuestas" y "una respuesta es de la narrativa de su
  padre" valen en práctica. Si el set final no los cumple, la pérdida es silenciosa: el validador solo avisa cuando
  el % de ruido, la cantidad de narrativas o el % de cuentas coordinadas caen fuera de [0,5×, 2×] de práctica.
  Hay que mirar esos avisos en `work/final/pipeline.log`.

## Código y librerías preexistentes

Todo el código de `qd/`, `herramientas/` y `tests/` se escribió durante el hackathon. Del repo del desafío usamos
`evaluacion/puntuar.py` (sin modificar) y la idea de normalización de `ejemplos/linea_base.py`.
Librerías públicas: DuckDB, pandas, NumPy, scikit-learn, networkx, sentence-transformers / PyTorch (modelo
`intfloat/multilingual-e5-small`), PyYAML, psutil, Plotly (reporte). LLM opcional vía API compatible con OpenAI.
Sin datos externos ni datos personales; sin claves en el repo.
