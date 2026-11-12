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

- **IA opcional:** exportá `LLM_API_KEY` y completá `llm.base_url` / `llm.modelo` en `config.yaml` (cualquier API
  compatible con OpenAI). Sin clave el pipeline corre igual, con reglas deterministas.
- **Reproducible:** semillas fijas en todo; las respuestas del LLM de la corrida final quedan en `cache/llm/final/`
  (commiteado), así el jurado reproduce la entrega **sin clave y sin costo**.
- **Calibración:** `make barrido P=<param> V=<v1,v2,...> ETAPA=<eNN>`; `python herramientas/diagnostico.py`.
- **Reporte para la demo:** `reportes/<set>.html` (se genera solo al final de cada corrida).
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
| `e08_coord` | co-acción repetida (misma publicación, ±Δ) ponderada contra virales, vs **modelo nulo por permutación** (fuente × período × banda de demora); + respuestas mutuas + hashtags raros; poda por triángulos; Louvain | compara contra lo esperable: virales, hiperactivos y seguidores de una fuente no son campañas |
| `e06_postura` | respuestas cortas acompañan, largas contradicen → **bipartición del grafo de respuestas** en bandos; orientación por LLM (2 llamadas por narrativa) o por votos (refutación explícita, negación, coordinación) | la postura sale de la estructura de la conversación, no de palabras clave |
| `e07_origen` | k-ésimo comentario más temprano entre los centrales, con confirmación de ráfaga; títulos por LLM | rango (no percentil): no se corre con el tamaño |
| `e09_exporta` + validador | 4 archivos de `entrega/`; falla si algo haría perder puntos en silencio | postura nunca vacía, sin `nan`, fechas ISO `-03:00` |

## Dónde se guardan los datos y por qué DuckDB + Parquet

- **DuckDB** (`work/<set>/qd.duckdb`) para las tablas intermedias: motor columnar embebido, cero operación, SQL
  completo, rápido para agregaciones y self-joins sobre cientos de miles de filas en una notebook.
- **Parquet particionado por día** (`work/<set>/parquet/comentarios/dia=N/`): formato abierto y columnar, legible por
  DuckDB, Spark, ClickHouse o BigQuery sin conversión. Elegimos no levantar un servidor (PostgreSQL/Elastic) porque
  para 150–200k comentarios agrega operación sin agregar capacidad.
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

## Costo (ver `entrega/costo.json`)

En la práctica (200.000 comentarios, CPU, 16 GB): ≈10 min en frío (dominado por embeddings), ≈35 s en caliente.
Con LLM: ~40 llamadas, ~30k tokens en total; el `usd_por_mil_comentarios` queda en milésimas de centavo.

## Resultados en el set de práctica (puntaje automático oficial)

| Corrida | Narrativas /30 | Coordinación /25 | Total /55 |
|---|---|---|---|
| Línea base del repo | — | — | ~17 |
| Nuestra solución, sin IA | 27,63 | 21,15 | **48,78** |
| Nuestra solución, con IA (orientación de bandos y títulos) | 28,22 | 21,15 | **49,37** |
| Pseudo-final sin IA (primeros 10 días, 75% de las publicaciones) | 27,89 | 16,32 | 44,21 |

B-cubed de narrativas 0,98 · ruido F1 0,985 · origen de emergentes 0,99 · coordinación F1 0,83.
El pseudo-final (`herramientas/pseudo_final.py`) mide generalización con menos datos que el set final real.

## Código y librerías preexistentes

Todo el código de `qd/`, `herramientas/` y `tests/` se escribió durante el hackathon. Del repo del desafío usamos
`evaluacion/puntuar.py` (sin modificar) y la idea de normalización de `ejemplos/linea_base.py`.
Librerías públicas: DuckDB, pandas, NumPy, scikit-learn, networkx, sentence-transformers / PyTorch (modelo
`intfloat/multilingual-e5-small`), PyYAML, psutil, Plotly (reporte). LLM opcional vía API compatible con OpenAI.
Sin datos externos ni datos personales; sin claves en el repo.
