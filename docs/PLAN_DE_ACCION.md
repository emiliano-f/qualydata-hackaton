# PLAN DE ACCIÓN · Desafío QualyData (narrativas y coordinación)

> Presupuesto: **20 h de reloj en total**, incluida la corrida sobre el set final.
> Fuente de verdad del puntaje: `evaluacion/puntuar.py`, leído línea por línea (ver §0).
> Análisis **A**: el propio. Análisis **B**: el adjunto. Lo que sigue es el arbitraje.

---

## 0. Las 8 reglas que impone `puntuar.py` (no negociables)

1. **Postura siempre completa.** El macro-F1 se calcula sobre *todos* los comentarios con narrativa real, **sin mirar la narrativa que predijimos**. Si un comentario lo marcamos como ruido y le dejamos la postura vacía, cuenta como falso negativo en las dos clases. Por eso `postura` ∈ {`apoya`,`rechaza`} en el 100% de las filas.
2. **El ruido es asimétrico.** B-cubed solo mira los comentarios con narrativa real, y cuando predecimos vacío lo trata como singleton (pierde recall). Meter ruido real dentro de un clúster no cuesta nada en B-cubed. Conclusión: **marcamos ruido solo con mucha confianza**.
3. **Origen por máxima superposición.** A cada narrativa emergente le toca el clúster predicho con **más comentarios en común** (conteo absoluto). Si una emergente queda absorbida por un clúster viejo, saca 0. Además, toda etiqueta usada en `comentarios.csv` tiene que figurar en `narrativas.csv`.
4. **Fechas:** `datetime.fromisoformat` en Python 3.9 o 3.10 no acepta `Z` ni nanosegundos. Una fecha **sin zona horaria tira `TypeError` y el script se cae**. Formato único: `%Y-%m-%dT%H:%M:%S-03:00`.
5. **`"nan"` NO se considera vacío** (los vacíos son `""`, `ninguna`, `none` y `null`). En `cuentas.csv` eso significa marcar como coordinadas a todas las cuentas genuinas. Siempre `fillna("")`.
6. **Coordinación:** cualquier grupo no vacío cuenta como "marcada". El B-cubed de grupos solo mira las cuentas realmente coordinadas, así que **los falsos positivos no lo afectan**. Una cuenta detectada que no podemos ubicar recibe `s_<cuenta_id>` (suma F1 y en B-cubed queda como singleton). **Fusionar dos campañas destruye la precisión.**
7. **IDs como texto** (`dtype=str, keep_default_na=False`). Un ID que falta cuenta como ruido, sin postura y como cuenta genuina.
8. **La métrica es B-cubed, no ARI ni Hungarian** (B lo suponía). No importa cuántos clústeres armemos; importa que lo que va junto quede junto.

Peso real de cada componente: coordinación-detección **17,5** · narrativas-B-cubed **15** · coordinación-grupos 7,5 · postura 6 · ruido 4,5 · origen 4,5.

---

## 1. Decisiones de arquitectura definitivas (arbitraje)

| Tema | A (propio) | B (adjunto) | **Ganador** | Por qué se descarta el otro |
|---|---|---|---|---|
| Almacenamiento | DuckDB + Parquet | DuckDB + Parquet | **DuckDB (`qd.duckdb`) + Parquet crudo particionado por día** | Coinciden. No vale la pena discutirlo. |
| Normalización y deduplicación | Hash normalizado + hash esqueleto | Regex en DuckDB + coseno > 0,95 | **A**, con `strip_accents()` de DuckDB | El regex de B no saca acentos ("está"≠"esta"), y el coseno sirve para casi-duplicados pero no reemplaza al hash esqueleto en el carril textual de coordinación. |
| Embeddings | e5-small local con caché | e5-small en GPU/ONNX con caché | **`intfloat/multilingual-e5-small`, prefijo `query: `, caché por hash**. Plan B: TF-IDF de caracteres + SVD | El "90 s" de B vale solo con GPU. En CPU tarda 3 a 6 min, y aceptarlo es más seguro que depender de que haya GPU el viernes. |
| Clustering | KMeans k≈150 + aglomerativo sobre centroides | Muestra estratificada por día + UMAP + HDBSCAN | **Híbrido: la muestra estratificada por día de B alimenta el MiniBatchKMeans de A; después, aglomerativo sobre los centroides** | UMAP+HDBSCAN suma 2 dependencias, no es determinista y tiene 4 hiperparámetros que interactúan. La muestra por día de B sí se queda: es lo que le da peso a las narrativas que nacen a mitad del período. |
| Deriva léxica y consolidación | Coseno + afinidad por respuestas + LLM solo en la zona gris (pares) | 1 prompt con ~60 micro-clústeres | **A** | El megaprompt de B no es determinista, hace fusiones transitivas sin control y puede tragarse una emergente (regla 0.3). Los pares acotados se pueden cachear y tienen un plan B (el umbral). |
| Ruido | Clasificador con atributos independientes del tema | Filtrar antes de la postura | **A + una marca del LLM por micro-clúster (`es_narrativa`)** | B calcula la postura solo sobre lo "no ruido" y pierde puntos (regla 0.1). |
| Postura | LLM por micro-clúster + léxico + respuestas | LLM en lotes de 50k "dudosos" mandando 15 palabras clave | **Cascada: respuestas → LLM por micro-clúster → léxico → LLM en lote solo para conflictos (tope 20k textos, texto completo truncado)** | Mandar solo palabras clave borra la negación, que es justo lo que define la postura. 50k sin tope agranda el riesgo de límites de tasa el viernes. |
| Herencia por `respuesta_a` | Reglas en orden cronológico | Inversión por negación | **A**, con marcadores aprendidos de práctica | Coinciden. A especifica el orden: el padre tiene que estar resuelto antes que el hijo. |
| `fecha_origen` | k-ésimo comentario más temprano entre los centrales (`conf` ≥ mediana) | Comentario #10 o percentil 3% después de superar X en 6 h | **A + la confirmación de ráfaga de B** | El percentil 3% de una narrativa de 5.000 comentarios es el comentario 150: puede caer horas tarde. Se usa rango, no percentil. |
| Coordinación: umbral | Modelo nulo (permutación) | K ≥ 3 publicaciones fijo | **A: el umbral sale del modelo nulo, con piso K ≥ 2** | K=3 es un número mágico calibrado para 14 días; el set final tiene 10. |
| Penalización anti-viral | 1/log(1+N) | 1/log(1+N_post) | **1/log(2+N_ventana)**, con N = cuentas distintas en esa publicación dentro de ±Δ | El volumen de toda la publicación castiga igual el minuto viral y el minuto tranquilo. |
| Frases copiadas | Hash esqueleto repetido por el par en ≥ 2 publicaciones | No entra por un único evento | **Ambos**, más un peso 1/log(2+cuentas que usan el hash) | Coinciden en que manda la repetición. |
| Partición | Louvain (networkx) | Leiden (igraph/leidenalg) | **`networkx.community.louvain_communities`** | leidenalg obliga a compilar o conseguir wheels, y eso es un riesgo en otra máquina. |
| Cuentas periféricas | Clasificador con atributos en percentiles | Propagar por lote de creación | **Ambos**: el clasificador marca y el lote de creación asigna el grupo | |
| Visualización | HTML con Plotly | HTML con Plotly (en `entrega/`) | **HTML con Plotly en `salida/<set>/reporte.html`** | Coinciden. `entrega/` queda con los 4 archivos exactos. |
| Hito del MVP | H8 | H6 | **H6** | Gana el gate más temprano: la línea base ya existe. |
| Corrida del set final | Ventana reservada H17 a H20 | No está (H19-20 es el video) | **A** | B no reserva tiempo para correr el set final, y las 20 h lo incluyen. |

---

## 2. Arquitectura del pipeline y contratos de datos

### 2.1 Árbol del repositorio

```
.
├── Makefile
├── config.yaml                 # TODOS los hiperparámetros, relativos (§4.2)
├── precios.yaml                # USD por millón de tokens (llenar con el precio vigente)
├── requirements.txt
├── .env.example                # LLM_API_KEY=  (.env va en .gitignore)
├── README.md
├── qd/
│   ├── __init__.py
│   ├── pipeline.py             # orquestador: --datos --salida --desde --sin-llm --embed
│   ├── db.py                   # conexión DuckDB, helpers SQL
│   ├── costo.py                # tracker (§2.4)
│   ├── llm.py                  # cliente + caché + interruptor de corte (§2.5)
│   ├── texto.py                # normalización, léxicos, marcadores
│   ├── etapas/
│   │   ├── e01_ingesta.py
│   │   ├── e02_embed.py
│   │   ├── e03_micro.py
│   │   ├── e04_narrativas.py   # consolidación + LLM zona gris + títulos
│   │   ├── e05_ruido.py
│   │   ├── e06_postura.py      # cascada + respuestas
│   │   ├── e07_origen.py
│   │   ├── e08_coord.py
│   │   ├── e09_exporta.py
│   │   └── e10_reporte.py
│   └── validar.py              # validador de entrega (§2.6)
├── modelos/                    # ruido.joblib, cuentas.joblib (entrenados con práctica, commiteados)
├── herramientas/
│   ├── barrido.py              # grillas en memoria usando puntuar.bcubed
│   ├── pseudo_final.py         # arma datos/p10 (§4.3)
│   ├── ensuciar.py             # arma datos/sucio para el test de robustez
│   └── entrenar.py             # entrena modelos/ con la verdad de práctica
├── evaluacion/puntuar.py       # el script oficial, sin tocar
├── ejemplos/linea_base.py
├── cache/                      # emb/<modelo>/{vectores.npy,indice.parquet}, llm/*.jsonl
├── work/<set>/qd.duckdb        # tablas intermedias (no se suben)
├── salida/<set>/               # corridas de práctica (no se suben)
├── experimentos.csv            # 1 fila por corrida: hash de config + puntajes
└── entrega/                    # SOLO los 4 archivos del set final + cache/llm de esa corrida
```

`requirements.txt` (con versiones fijadas, compatibles con Python 3.9):
`duckdb>=1.0`, `pandas>=1.5`, `pyarrow>=12`, `numpy<2.1`, `scikit-learn>=1.3,<1.6`, `sentence-transformers>=2.7`, `networkx>=3.1`, `openai>=1.30`, `tenacity>=8`, `pyyaml`, `plotly>=5`, `psutil`, `joblib`.

### 2.2 `Makefile` (orquestación de un solo comando)

```make
SET    ?= practica
DATOS  ?= datos/$(SET)
SALIDA ?= salida/$(SET)
PY     ?= python
FLAGS  ?=

run:
	$(PY) -m qd.pipeline --datos $(DATOS) --salida $(SALIDA) --config config.yaml $(FLAGS)

score: run
	$(PY) evaluacion/puntuar.py --verdad $(DATOS)/verdad --entrega $(SALIDA) --json $(SALIDA)/puntaje.json

desde:          # make desde ETAPA=e04  -> recalcula desde esa etapa (loop en caliente)
	$(PY) -m qd.pipeline --datos $(DATOS) --salida $(SALIDA) --config config.yaml --desde $(ETAPA) $(FLAGS)
	$(PY) evaluacion/puntuar.py --verdad $(DATOS)/verdad --entrega $(SALIDA) --json $(SALIDA)/puntaje.json

sinllm:         # camino determinista conocido como bueno
	$(MAKE) run FLAGS="--sin-llm"

final:
	$(MAKE) run SET=final SALIDA=entrega
	$(PY) -m qd.validar --datos datos/final --entrega entrega

test-sucio:
	$(PY) herramientas/ensuciar.py --origen datos/practica --destino datos/sucio
	$(MAKE) run SET=sucio FLAGS="--sin-llm"
```

`pipeline.py` corre las etapas en orden. Cada etapa lee y escribe **tablas con nombre fijo** en `work/<set>/qd.duckdb`, así que `--desde eNN` saltea las anteriores si sus tablas existen. **El validador corre siempre al final**: si falla, el proceso sale con código ≠ 0.

### 2.3 Esquema de tablas (DuckDB, `work/<set>/qd.duckdb`)

```sql
-- e01_ingesta -------------------------------------------------------------
-- Se lee con all_varchar=true y se castea con try_cast: los datos sucios NO rompen la ingesta
com(            -- 1 fila por comentario válido
  comentario_id VARCHAR PRIMARY KEY, publicacion_id VARCHAR, cuenta_id VARCHAR,
  fecha TIMESTAMPTZ, ts BIGINT,                 -- epoch en segundos
  dia INTEGER,                                  -- días desde el mínimo del set
  texto VARCHAR, respuesta_a VARCHAR,           -- NULL si viene vacío o no existe en com
  me_gusta INTEGER, norm VARCHAR, n_tok INTEGER,
  h_texto UBIGINT,                              -- hash(norm)
  h_esq UBIGINT                                 -- hash(tokens ≥3 letras, sin stopwords, ordenados)
);
com_descartados(comentario_id VARCHAR, motivo VARCHAR)  -- fecha inválida, ID duplicado, etc.
                                                         -- igual SALEN en comentarios.csv (como ruido)
pub(publicacion_id VARCHAR, fuente_id VARCHAR, fecha TIMESTAMPTZ, titular VARCHAR)
cta(cuenta_id VARCHAR, fecha_creacion TIMESTAMPTZ, seguidores INTEGER, siguiendo INTEGER,
    edad_dias DOUBLE)                           -- respecto del primer comentario del set
textos(h_texto UBIGINT PRIMARY KEY, norm VARCHAR, n INTEGER, dia_min INTEGER, fila_emb INTEGER)
-- Copia para el relato de escala: COPY com TO 'work/<set>/parquet' (FORMAT parquet, PARTITION_BY (dia))

-- e03_micro / e04_narrativas ----------------------------------------------
micro_txt(h_texto UBIGINT, micro INTEGER, conf DOUBLE)  -- conf = coseno con su centroide
micro(micro INTEGER, n INTEGER, narrativa VARCHAR, es_narr_llm BOOLEAN,
      afirmacion_llm VARCHAR, postura_llm VARCHAR)        -- 'apoya'|'rechaza'|'mixta'|NULL
narr(narrativa VARCHAR, titulo VARCHAR, frases_clave VARCHAR, afirmacion VARCHAR,
     n INTEGER, fecha_origen TIMESTAMPTZ)

-- e05..e07 ----------------------------------------------------------------
asig(comentario_id VARCHAR, micro INTEGER, narrativa VARCHAR,  -- NULL = ruido
     conf DOUBLE, p_ruido DOUBLE, heredada BOOLEAN,
     postura VARCHAR, fuente_postura VARCHAR)  -- 'resp'|'llm_micro'|'lex'|'llm_lote'|'default'

-- e08_coord ----------------------------------------------------------------
ev(cuenta_id VARCHAR, publicacion_id VARCHAR, ts BIGINT, n_ventana INTEGER)
pares(u VARCHAR, v VARCHAR, n_pub INTEGER, s_t DOUBLE, n_txt INTEGER, s_txt DOUBLE, w DOUBLE)
pares_nulo(...)                                 -- mismo esquema, sobre el modelo nulo
feat_cta(cuenta_id VARCHAR, ...atributos en percentiles..., p_coord DOUBLE)
grupos(cuenta_id VARCHAR, grupo VARCHAR)        -- '' | 'g001' | 's_<cuenta_id>'
```

Caché de embeddings, compartida entre sets: `cache/emb/<modelo>/vectores.npy` (float16, append-only) + `indice.parquet(sha1_norm, fila)`. La clave es el **sha1 del texto normalizado**, no el `hash()` de DuckDB (que puede cambiar entre versiones).

**Tiempos (a medir en H1 y anotar acá):**

| Etapa | Frío, CPU (8 núcleos) | Frío, GPU T4 | Caliente (caché) |
|---|---|---|---|
| e01 ingesta + normalización | 10 a 20 s | igual | se saltea |
| e02 embeddings (textos únicos) | **3 a 6 min** (ONNX: 1,5 a 3) | 40 a 90 s | **0 s** |
| e03 + e04 micro-clusters + unión | 20 a 40 s | igual | 20 a 40 s |
| LLM (todas las pasadas, 16 en paralelo) | 1 a 3 min | igual | **0 s (caché)** |
| e05..e07 | menos de 15 s | igual | menos de 15 s |
| e08 coordinación + nulo | 30 a 60 s | igual | 30 a 60 s (o `--desde e09`) |
| e09 + e10 + validación | menos de 10 s | igual | menos de 10 s |
| **Total** | **≈6 a 11 min** | **≈3 a 4 min** | **≈25 a 60 s** (`--desde e04`: menos de 30 s) |

> Hay que ser honestos: **una corrida en frío de menos de 3 minutos solo se logra con GPU** o con el embedding del plan B. En CPU tarda entre 6 y 11 minutos, y es aceptable porque en frío corre dos veces en todo el evento (práctica y final). **Lo que importa es el loop en caliente**, y ese baja de 30 s.

### 2.4 Tracker de `costo.json` (`qd/costo.py`)

```python
class Costo:
    def __init__(s, n_comentarios, cfg): s.t0=time.perf_counter(); s.etapas={}; s.tok_in=s.tok_out=0; s.llamadas=0; s.modo="completo"
    @contextmanager
    def etapa(s, nombre): t=time.perf_counter(); yield; s.etapas[nombre]=round(time.perf_counter()-t,1)
    def llm(s, usage, desde_cache): # solo se suman los tokens que NO salieron de la caché
        if not desde_cache: s.tok_in+=usage.prompt_tokens; s.tok_out+=usage.completion_tokens; s.llamadas+=1
    def escribir(s, ruta):
        usd = s.tok_in/1e6*P["in"] + s.tok_out/1e6*P["out"]          # de precios.yaml
        json.dump({"comentarios_procesados": n, "segundos": round(time.perf_counter()-s.t0),
                   "tokens_ia": s.tok_in+s.tok_out, "usd_ia": round(usd,4),
                   "usd_por_mil_comentarios": round(usd/n*1000,5),
                   "hardware": f"{platform.processor() or platform.machine()}, {os.cpu_count()} núcleos, "
                               f"{psutil.virtual_memory().total//2**30} GB RAM, GPU: {gpu or 'no'}",
                   "almacenamiento": "DuckDB local + Parquet particionado por día",
                   "notas": f"embeddings locales {modelo}; LLM solo por clúster y en casos dudosos; modo_llm={s.modo}",
                   "desglose_segundos": s.etapas, "llamadas_llm": s.llamadas, "modo_llm": s.modo}, ...)
```

- Lo instancia `pipeline.py`; cada etapa corre dentro de `with costo.etapa("e02_embed"):`.
- **En la corrida final se borra la caché del LLM antes de arrancar** (`rm -rf cache/llm/final`), así el costo reportado es real. Después se commitea esa caché para que el jurado pueda reproducir sin clave.

### 2.5 Cliente de LLM y estrategia de respaldo (`qd/llm.py`)

- **Un solo cliente**: el SDK `openai` con `base_url` configurable. Sirve para Gemini por su endpoint compatible con OpenAI (`https://generativelanguage.googleapis.com/v1beta/openai/`) y para cualquier proveedor compatible. El modelo y la URL van en `config.yaml`; la clave se lee de la variable de entorno `LLM_API_KEY` y nunca entra al repo.
- `temperature=0`, `response_format={"type":"json_object"}`, `max_tokens` acotado por prompt.
- **Caché**: `cache/llm/<set>/<sha256(modelo+prompt)>.json`. Un acierto de caché no gasta tokens y no hace llamada.
- **Concurrencia**: `ThreadPoolExecutor(max_workers=cfg.llm.paralelo=16)`, con un tiempo límite de 30 s por llamada.
- **Reintentos**: `tenacity` con `stop_after_attempt(3)` y `wait_exponential(1, 2, 20)`, solo ante errores 429, 5xx o tiempo agotado.
- **Interruptor de corte**: con 8 fallas seguidas, o si los tokens superan `cfg.llm.tope_tokens=2_000_000`, se pasa a `modo="degradado"` y **todas las llamadas que siguen devuelven `None`** sin tocar la red.
- **Validación de la respuesta**: JSON parseable con las claves esperadas y valores dentro del dominio. Si no, se toma como `None` para ese ítem.
- **Cada uso del LLM tiene un respaldo determinista** que se activa cuando la respuesta es `None`:

| Uso | Respaldo |
|---|---|
| `es_narrativa` del micro-clúster | Clasificador de ruido solo |
| Unión en la zona gris | No se une (conservador) |
| Título y frases clave | Top 6 términos TF-IDF del clúster contra el resto |
| Postura del micro-clúster | Voto mayoritario del léxico |
| Postura en lote | Léxico, y si no hay señal, la postura mayoritaria de la narrativa |

- Sin `LLM_API_KEY`, `--sin-llm` se activa solo. **`make run` funciona sin clave**: el jurado puede volver a correrlo.

### 2.6 Validador (`qd/validar.py`, corre dentro de `make run`)

Si alguna de estas condiciones falla, sale con código 1:

1. Los 4 archivos existen y tienen las columnas exactas.
2. El conjunto de `comentario_id` es igual al de la entrada (incluidos los descartados) y no hay duplicados.
3. `postura` ∈ {apoya, rechaza} en el **100%** de las filas.
4. Ninguna celda contiene `nan`, `NaN`, `None` ni `NaT`.
5. Toda etiqueta de `comentarios.csv` está en `narrativas.csv`.
6. Toda `fecha_origen` matchea `^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d-03:00$` y `datetime.fromisoformat()` la acepta.
7. `cuentas.csv` cubre la unión de las cuentas de `cuentas.csv.gz` y de `comentarios`.
8. `costo.json` tiene las 8 claves del enunciado.
9. **Chequeo de sanidad, solo aviso**: el % de ruido, la cantidad de narrativas y el % de cuentas coordinadas caen dentro de [0,5×, 2×] de lo visto en práctica.

---

## 3. Receta algorítmica paso a paso

### Módulo A: narrativas, ruido, postura y origen

**A1. Ingesta (`e01`)**
```sql
CREATE TABLE raw AS SELECT * FROM read_csv('{d}/comentarios.csv.gz', header=true, all_varchar=true, quote='"', escape='"');
-- dedup por ID (se queda la primera aparición), try_cast de fecha y me_gusta, respuesta_a se anula si no existe
CREATE TABLE com AS
WITH r AS (SELECT *, row_number() OVER (PARTITION BY comentario_id) rn FROM raw WHERE comentario_id IS NOT NULL)
SELECT comentario_id, publicacion_id, cuenta_id,
       try_cast(fecha AS TIMESTAMPTZ) fecha, epoch(try_cast(fecha AS TIMESTAMPTZ))::BIGINT ts,
       coalesce(texto,'') texto, nullif(trim(respuesta_a),'') respuesta_a,
       coalesce(try_cast(me_gusta AS INTEGER),0) me_gusta
FROM r WHERE rn=1;
-- las filas con fecha NULL van a com_descartados; después se ponen respuesta_a=NULL los que no están en com
```
La normalización se hace en SQL: `regexp_replace(regexp_replace(lower(strip_accents(texto)), '@\w+|#\w+|https?://\S+', ' ', 'g'), '[^a-z0-9ñ ]', ' ', 'g')`. Después se colapsan las letras repetidas (`(\w)\1{2,}`→`\1`), y en Python se expanden las abreviaturas de `linea_base.ABBREVIATIONS`, pero **solo sobre `textos` (únicos)**.

**A2. Embeddings (`e02`)**
- `SentenceTransformer("intfloat/multilingual-e5-small")`, con `max_seq_length=96`, `encode(["query: "+t ...], batch_size=256, normalize_embeddings=True)`, solo sobre los `textos` que faltan en caché.
- Si `torch.cuda.is_available()`, va a GPU. Si existe `backend="onnx"` (sentence-transformers ≥ 3.2), se usa.
- **Plan B (`--embed tfidf`)**: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, max_features=200_000, sublinear_tf=True)`, luego `TruncatedSVD(256, random_state=42)` y normalización L2. Tarda unos 40 s. Se prueba una vez en H2.

**A3. Micro-clusters (`e03`)**
- Muestra de ajuste **estratificada por `dia_min`** (lo que aporta B): hasta `cfg.micro.por_dia=3000` textos únicos por día, con semilla 42.
- `MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=42)`, con `k = clip(round(sqrt(n_unicos)/2), 80, 250)` y `sample_weight = log1p(n)` sobre la muestra. Después se aplica `predict` a todos los únicos y `conf = coseno(x, centroide)`.
- Se escribe `micro_txt`. El resultado se cachea por `(hash_config_micro, hash_textos)`.

**A4. Consolidación en narrativas (`e04`)**

Esta pieza es el control de la granularidad.

1. `C` = centroides normalizados. `d_nn` = distancia de cada centroide a su vecino más cercano.
   `T = quantile(d_nn, cfg.narr.q_T)`, con `q_T` inicial 0,60 y grilla 0,3 a 0,9. Es un umbral **relativo** al set.
2. `AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=T).fit(C)`.
3. **Afinidad por respuestas**: `R_ij` = (respuestas de *i*→*j* + *j*→*i*) / min(n_i, n_j). Se unen las narrativas cuando `R_ij ≥ cfg.narr.r_resp` (0,15) **y** `dist ≤ T + cfg.narr.delta_gris`. Así se unen los subgrupos que apoyan y rechazan.
4. **Zona gris con LLM**: los pares de narrativas con `T < dist ≤ T + delta_gris` (inicial 0,06), ordenados por distancia y con un tope de `cfg.llm.max_pares=60`, pasan por el prompt P2. Se unen si `misma=true`, usando union-find.
5. Las etiquetas se renumeran `n001…` por tamaño. Los micro-clústeres que P1 marcó con `es_narrativa=false` y se quedaron en su propia narrativa pasan a **candidatos a ruido** (ver A5).

**A5. Ruido (`e05`)**

- **Atributos, todos independientes del tema**:
  - `n_tok`, `len_chars`, `es_resp`, `frac_emoji_o_simbolos`, `tiene_mencion`
  - `conf`, el percentil de `conf` dentro de su narrativa, `log(n_narrativa)`, `log(n_micro)`
  - `es_narr_llm` (NaN si no hay LLM), `frac_stopwords`
- **Modelo**: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, class_weight="balanced", random_state=42)`, entrenado con la verdad de práctica en `herramientas/entrenar.py` y guardado en `modelos/ruido.joblib`.
- **Regla**: es ruido si `p_ruido ≥ cfg.ruido.tau` (inicial 0,75, alto por la regla 0.2) **y** no es una respuesta corta con padre en una narrativa. Las respuestas cortas heredan siempre.
- **Plan B sin modelo**: es ruido si `n_tok ≤ 3` y no es respuesta, o si el micro-clúster es chico (`n_micro < 0,05%` de los comentarios) y tiene `conf` en el percentil 10 o menos.

**A6. Postura (`e06`): cascada en este orden**

0. **Antes, sobre práctica (H3)**: `herramientas/entrenar.py` extrae los marcadores. Toma las respuestas con `n_tok ≤ 6`, compara la postura de la verdad del hijo contra la del padre, se queda con los n-gramas que predicen *igual* o *inversa*, y conserva solo los que aparecen en ≥ 3 narrativas distintas. El resultado va a `qd/texto.py` como `ACUERDO` y `DESACUERDO`.
   - Semillas iniciales de ACUERDO: tal cual, exacto, totalmente, coincido, asi es, posta, obvio, ni hablar, 100.
   - Semillas iniciales de DESACUERDO: mentira, falso, fake, verso, chamuyo, nada que ver, no es asi, cualquiera, inventan, desmintio, sin pruebas, trucho.
   - `REJECT_MARKERS` sale de la línea base, ampliado con lo que aparezca en práctica.
1. **Respuestas** (orden cronológico, el padre ya resuelto):
   - si `n_tok ≤ cfg.resp.corta` (inicial 5) y tiene un marcador de ACUERDO o ningún marcador: hereda narrativa y postura (`fuente='resp'`, `heredada=true`);
   - si tiene un marcador de DESACUERDO: hereda la narrativa con la postura **invertida**;
   - si es larga y su `conf` está por debajo de la mediana de su micro-clúster: hereda solo la narrativa y la postura se decide en los pasos que siguen;
   - si el padre es ruido, no hereda nada.
2. **Micro-clúster con LLM (P3)**: si P3 devuelve `apoya` o `rechaza` y ≥ 75% de las etiquetas por ejemplo coinciden, esa postura se asigna a los comentarios del micro-clúster **que no tienen un marcador contrario**.
3. **Léxico**: con un marcador de rechazo, `rechaza`. En un micro-clúster `rechaza` sin marcador, `rechaza`.
4. **LLM en lote (P4)**: va a los conflictos (léxico contra micro-clúster), a los micro-clústeres `mixta` y a lo que no tuvo señal. Se deduplica por `h_texto`, se ordena por cantidad de veces que aparece el texto (descendente), con un tope de `cfg.llm.max_lote=20000` textos y lotes de 40.
5. **Por defecto**: la postura mayoritaria de su narrativa. Para el ruido, `apoya` (regla 0.1: nunca vacío).

**A7. `fecha_origen` (`e07`)**, por cada narrativa predicha:
```
core = comentarios de la narrativa con heredada=false y conf ≥ mediana(conf de la narrativa), ordenados por fecha
for i, c in enumerate(core):
    if count(core en [c.fecha, c.fecha + 6h]) ≥ m:   # m = cfg.origen.m (inicial 3): confirmación de ráfaga
        candidato = core[i + k - 1]                   # k = cfg.origen.k (inicial 1; grilla 1..8)
        break
fecha_origen = floor_minuto(candidato).astimezone(timezone(timedelta(hours=-3))).strftime("%Y-%m-%dT%H:%M:%S-03:00")
```
Si no hay ráfaga confirmada, se usa el k-ésimo de `core`. La grilla (k, m) se calibra contra `origen_por_narrativa` del script.

**A8. Exportación (`e09`)**
- `comentarios.csv`: los IDs de `com` más los de `com_descartados`. A los descartados les corresponde `narrativa=""` y `postura="apoya"`.
- `narrativas.csv`: `titulo` y `frases_clave` de P5 o del respaldo TF-IDF, unidos con `|`, sin saltos de línea.

**Prompts (español, salida JSON, `temperature=0`)**

```
P1 · micro-clúster (1 llamada por micro, unas 150 en total; 12 ejemplos estratificados por conf)
SISTEMA: Analizás comentarios de redes sobre un tema público. Respondé solo con JSON.
USUARIO: Estos 12 comentarios fueron agrupados por similitud:
1. «...»  … 12. «...»
¿El grupo expresa una AFIRMACIÓN o ENCUADRE sobre el tema (aunque algunos la nieguen),
o es ruido (reacciones, saludos, menciones, chistes, fuera de tema)?
{"es_narrativa": true|false, "afirmacion": "<la afirmación en forma POSITIVA, ≤20 palabras>"}

P2 · zona gris (1 llamada por par, ≤60)
USUARIO: Grupo A: afirmación «...»; ejemplos: «...»×5
Grupo B: afirmación «...»; ejemplos: «...»×5
¿A y B sostienen o discuten LA MISMA afirmación de fondo (aunque con otras palabras, en otro
momento o con postura opuesta)? Si una es un hecho o rumor nuevo distinto, respondé false.
{"misma": true|false}

P3 · postura del micro-clúster respecto de la narrativa (después de A4; ≤150 llamadas)
USUARIO: Afirmación de referencia: «<afirmacion canónica de la narrativa>».
Para cada comentario indicá si APOYA (afirma, respalda, amplifica) o RECHAZA (niega, refuta,
ridiculiza) la afirmación. "Es mentira que X" RECHAZA X.
1. «...» … 12. «...»
{"posturas": ["A"|"R", ... 12], "mayoritaria": "apoya"|"rechaza"|"mixta"}

P4 · lote de dudosos (≤500 llamadas; 40 textos COMPLETOS truncados a 220 caracteres)
USUARIO: Afirmación de referencia por id de narrativa: {"n003":"...", "n007":"..."}
Para cada ítem [id, narrativa, texto] devolvé A (apoya) o R (rechaza) respecto de SU narrativa.
{"r": {"<id>": "A"|"R", ...}}

P5 · nombre de la narrativa (1 por narrativa, unas 20; recibe las afirmaciones de sus micro-clústeres + 8 ejemplos)
{"afirmacion_canonica": "...", "titulo": "<≤12 palabras>", "frases_clave": ["...", ... 6]}
```
Orden de las llamadas: P1, después A4 con P2, después P5 (que fija la afirmación canónica), después P3 y por último P4. Tokens estimados: menos de 1,5 M en total.

### Módulo B: coordinación (`e08`)

**B1. Eventos con volumen de la ventana** (Δ = `cfg.coord.delta_s`, inicial 180; grilla 60, 180, 300, 600)
```sql
CREATE TABLE ev AS
WITH e AS (SELECT cuenta_id, publicacion_id, min(ts) ts FROM com GROUP BY 1,2)
SELECT *, count(*) OVER (PARTITION BY publicacion_id ORDER BY ts
          RANGE BETWEEN {D} PRECEDING AND {D} FOLLOWING) AS n_ventana
FROM e;
```

**B2. Carril temporal** (penaliza los momentos virales)
```sql
CREATE TABLE pares_t AS
SELECT a.cuenta_id u, b.cuenta_id v,
       count(DISTINCT a.publicacion_id)                              AS n_pub,
       sum(1.0 / ln(2 + greatest(a.n_ventana, b.n_ventana)))         AS s_t
FROM ev a JOIN ev b
  ON a.publicacion_id = b.publicacion_id AND a.cuenta_id < b.cuenta_id
 AND b.ts BETWEEN a.ts - {D} AND a.ts + {D}
GROUP BY 1,2 HAVING count(DISTINCT a.publicacion_id) >= 2;
```
Una publicación viral produce muchos pares con `n_pub=1`, que quedan afuera por el `HAVING`. Lo que sobrevive es la repetición del mismo par.

**B3. Carril textual** (penaliza las frases copiadas)
```sql
CREATE TABLE h_pop AS SELECT h_esq, count(DISTINCT cuenta_id) n_ctas FROM com WHERE n_tok >= 6 GROUP BY 1;
CREATE TABLE pares_x AS
SELECT a.cuenta_id u, b.cuenta_id v,
       count(DISTINCT a.publicacion_id || '|' || b.publicacion_id) n_txt,
       sum(1.0 / ln(2 + h.n_ctas)) s_txt
FROM com a JOIN com b ON a.h_esq = b.h_esq AND a.cuenta_id < b.cuenta_id
JOIN h_pop h ON h.h_esq = a.h_esq
WHERE a.n_tok >= 6 AND h.n_ctas BETWEEN 2 AND {MAX_CTAS_HASH}   -- tope = percentil 99,5 de n_ctas
GROUP BY 1,2 HAVING n_txt >= 2;
```

**B4. Modelo nulo y umbrales relativos**
- Se permuta `cuenta_id` **dentro de cada `publicacion_id`** con `np.random.default_rng(42)`. Esto conserva el perfil temporal y textual de cada publicación (la viralidad queda intacta) y rompe la identidad de las cuentas. Después se recalculan B1 a B3, lo que da `pares_nulo`.
- Para cada carril c ∈ {t, x}: `θ_c = min s` tal que `#nulo(s_c ≥ s) / #obs(s_c ≥ s) ≤ cfg.coord.alfa` (inicial 0,01, es decir 1% de falsos descubrimientos esperados).
- Aristas: `w = s_t/θ_t + s_txt/θ_x`, conservando las que tienen `s_t ≥ θ_t` o `s_txt ≥ θ_x`.

**B5. Atributos por cuenta** (todos en **percentil dentro del set**, `rank(pct=True)`):
- del grafo: `grado`, `w_max`, `w_sum`;
- de actividad: fracción de sus eventos con un compañero dentro de ±Δ, `comentarios_por_dia`, entropía de sus narrativas, fracción de textos cuyo `h_esq` usa otra cuenta, fracción de respuestas, `me_gusta` medio;
- de la cuenta: `edad_dias`, `log1p(seguidores)`, `siguiendo/(1+seguidores)`;
- **tamaño del lote de creación**: z = (cuentas creadas en la misma hora − media)/desvío, calculado sobre la serie horaria de `fecha_creacion`.

**B6. Clasificación**
- `HistGradientBoostingClassifier` (mismos hiperparámetros que A5) → `modelos/cuentas.joblib`. Se valida dejando **una campaña afuera** por vez.
- **Una cuenta queda marcada** si `p_coord ≥ cfg.coord.tau` (inicial: el valor que maximiza F1 en esa validación) **o** si tiene 2 o más aristas fuertes (w ≥ 2). La unión protege contra estilos de campaña nuevos el viernes.
- Plan B sin modelo: solo la regla de las aristas.

**B7. Partición → `cuentas.csv`**
1. Subgrafo de las cuentas marcadas, con peso `w`. Se aplica `nx.community.louvain_communities(G, weight="w", resolution=cfg.coord.gamma, seed=42)`, con γ inicial 1,0 y grilla 0,5 a 2.
2. Comunidades de menos de 3 cuentas: se suman a la comunidad vecina con mayor peso total. Si no tienen vecinos, se asignan al grupo de su lote de creación cuando el lote tiene ≥ 3 cuentas y ≥ 50% de ellas marcadas. Si nada aplica, `s_<cuenta_id>`.
3. Las etiquetas son `g001…` por tamaño. Las cuentas genuinas llevan `""`. El archivo cubre todas las cuentas (validador, punto 7).

---

## 4. Plan de calibración contra el script (anti-overfitting)

### 4.1 Qué mirar en la hora 1 (sobre `datos/practica/verdad/`)

Todo se anota en `README.md › Hallazgos`:

**Narrativas**
- Cantidad, distribución de tamaños y % de ruido.
- % de comentarios que rechazan, por narrativa (para saber si el macro-F1 está desbalanceado).
- Cuántas son emergentes y **cómo se relaciona su `fecha_origen` real con el mínimo de las fechas de sus comentarios** (si coincide, k=1 es lo correcto).
- % de respuestas, y % de respuestas cortas que heredan.

**Coordinación**
- Cantidad de campañas, tamaño y % de cuentas coordinadas.
- Por campaña: tasa de texto duplicado, mediana del Δt entre cuentas en la misma publicación, publicaciones por cuenta, z del lote de creación y edad.

**Trampas**
- Las 5 publicaciones con más comentarios por minuto: ¿qué % de esas cuentas es genuino?
- Los `h_esq` usados por más de 50 cuentas: ¿qué % es genuino?

**Base**
- Puntaje de la línea base y de la línea base con postura completa (regla 0.1).

### 4.2 Hiperparámetros: todos en `config.yaml`, todos relativos

| Parámetro | Definición | Inicial | Grilla |
|---|---|---|---|
| `micro.k` | clip(√n_únicos/2, 80, 250) | auto | factor 0,5 / 1 / 1,5 |
| `micro.por_dia` | únicos por día en la muestra | 3000 | fijo |
| `narr.q_T` | cuantil de d_nn entre centroides | 0,60 | 0,30 a 0,90 de a 0,05 |
| `narr.delta_gris` | ancho de la zona gris | 0,06 | 0,03 / 0,06 / 0,10 |
| `narr.r_resp` | afinidad por respuestas | 0,15 | 0,05 a 0,30 |
| `ruido.tau` | p_ruido mínima | 0,75 | 0,5 a 0,95 |
| `resp.corta` | tokens máximos de una respuesta "corta" | 5 | 3 a 8 |
| `origen.k`, `origen.m` | rango y confirmación de ráfaga | 1, 3 | k 1 a 8, m 1 a 5 |
| `coord.delta_s` | ventana de co-acción | 180 | 60 / 180 / 300 / 600 |
| `coord.alfa` | FDR contra el nulo | 0,01 | 0,001 a 0,05 |
| `coord.tau` | p_coord mínima | la de la validación por campaña | — |
| `coord.gamma` | resolución de Louvain | 1,0 | 0,5 a 2,0 |

**Barridos en memoria** (`herramientas/barrido.py`): `from evaluacion.puntuar import bcubed, f1_binary`. Para la granularidad se barre q_T con todo lo demás congelado (el aglomerativo sobre unos 150 centroides tarda milisegundos), unas 13 configuraciones en menos de 20 s. **La función objetivo es el puntaje combinado de narrativas** (0,5·B³ + 0,15·ruido + 0,2·postura + 0,15·origen) y no B-cubed solo, por la regla 0.3.

### 4.3 Protocolo anti-overfitting

1. **Pseudo-final P10** (`herramientas/pseudo_final.py`): días 1 a 10 y el 75% de las publicaciones, con su verdad filtrada. Simula 150k comentarios en 10 días.
2. **Validación dejando una campaña afuera** para `cuentas.joblib`, y **mitades de narrativas** (se calibra q_T con una mitad y se evalúa con la otra).
3. **Estabilidad**: 3 semillas. Si el puntaje varía más de 1 punto, el parámetro es frágil y se prefiere el valor central de la meseta, no el pico.
4. **Regla de aceptación**: un cambio entra solo si sube el puntaje en práctica completa **y** no baja más de 1 punto en P10. Todo se registra en `experimentos.csv` (`ts, git_sha, hash_config, narr, coord, total, total_p10`).
5. **Los modelos entrenados (ruido y cuentas) solo usan atributos en percentil o independientes del tema.** Prohibido usar n-gramas de contenido como atributos.

---

## 5. Runbook de 20 horas (checklist operativo con Go / No-Go)

Carriles: **A** = narrativas · **B** = coordinación · **C** = infraestructura, visualización y pitch. Si trabajás solo, seguí el orden de las filas y hacé primero lo de A y B.

| Hora | A | B | C | **Gate (Go / No-Go) → contingencia** |
|---|---|---|---|---|
| **H0 a H1** | Línea base + `puntuar`; parche de "postura siempre completa"; análisis de la §4.1 | Análisis exploratorio de campañas y trampas (§4.1) | Repo, `requirements`, `Makefile`, `config.yaml`, `e01_ingesta`. **Lanzar `e02` en segundo plano** | **G1:** ¿el embedding avanza a ≥ 150 textos/s? Si no, `--embed tfidf` como principal y e5 queda para la noche |
| H1 a H3 | `e03` + `e04` sin LLM; barrido de q_T | B1 a B4 (pares + nulo + θ) | `costo.py`, `validar.py`, esqueleto de `e09` | **G2:** ¿el B-cubed de narrativas supera al TF-IDF+k-means de la línea base? Si no, revisar el prefijo `query: ` y la normalización; si sigue igual, usar el plan B de embeddings |
| H3 a H5 | Marcadores aprendidos; `e06` (respuestas + léxico); `e07` | B5, B7 con regla de aristas (sin modelo) | `herramientas/entrenar.py` + `ruido.joblib`, `cuentas.joblib` | — |
| H5 a H6 | `e05` ruido, integración | B6 clasificador + validación por campaña | `make run` de punta a punta en un **venv limpio** | **G3 (H6) MVP:** `make run` genera los 4 archivos, pasa el validador y el total automático **supera a la línea base por ≥ 8 puntos**. Go: etiqueta `v0-mvp`. No-Go: H6 a H8 solo para que corra; se elimina todo lo que no anda y se va con reglas puras |
| H6 a H8 | `llm.py` + P1 + P5 (títulos) | Ajuste de Δ, α, γ | Caché del LLM, interruptor de corte, `--sin-llm` probado | — |
| H8 a H10 | P2 (zona gris) + P3 + P4 | Asignación de periféricas por lote de creación | `herramientas/pseudo_final.py`, `experimentos.csv` | **G4 (H10):** ¿la capa de LLM sube la postura ≥ 0,05 de macro-F1 o el combinado de narrativas ≥ 1 punto, **y** no falla más de 2% de las llamadas? Si no, `llm.activo=false` por defecto (queda solo P5 para los títulos) y no se gasta una hora más en eso |
| H10 a H13 | Calibración (§4.2) + P10 + estabilidad | Lo mismo para coordinación | `ensuciar.py` + `make test-sucio`; reproducibilidad (2 corridas dan el mismo sha256 de salida) | **G5 (H13):** ¿P10 cae ≤ 15% contra práctica completa? Si no, volver al último parámetro relativo estable; ningún umbral absoluto sobrevive |
| H13 a H15 | Revisar el top 10 de errores de postura y ruido (sin tocar umbrales) | Revisar los falsos positivos de las trampas virales | `e10_reporte.py`: (1) área apilada diaria por narrativa con la marca de origen y la proporción apoya/rechaza, (2) red de operaciones con Plotly (nodos coloreados por grupo, tamaño por actividad), (3) tabla de narrativas + `costo.json`. README: cómo correrlo, por qué DuckDB, escala, código preexistente | — |
| **H15 a H16** | **Congelamiento del código.** `make score` en frío para cronometrarlo y anotarlo; video de demo con datos de práctica; ensayo de 1 minuto del pitch | | | **G6 (H16):** `git tag v1-final-candidato`. Cualquier cambio posterior requiere que mejore práctica **y** P10, y que pase `make test-sucio` |
| H16 a H17 | Margen y descanso. No se programa | | | |
| **H17 a H20** | **Ventana del set final** (se adelanta si se publica antes) | | | Ver la checklist de abajo |

### Checklist de la ventana final (en este orden)

- [ ] `T-0`: descargar en `datos/final/` y verificar los 4 archivos con `zcat | head` (columnas iguales a práctica).
- [ ] `rm -rf cache/llm/final entrega/*`, exportar `LLM_API_KEY` y correr `make final`.
- [ ] El validador pasa. Revisar los avisos de sanidad: % de ruido, cantidad de narrativas y % de cuentas coordinadas dentro de [0,5×, 2×] de práctica.
- [ ] Mirar a ojo `narrativas.csv`: ¿los títulos tienen sentido? ¿Hay fechas de origen a mitad del período?
- [ ] Si `make final` falla o la sanidad da valores absurdos: `make final FLAGS="--sin-llm"`, que es el camino determinista probado en H10 a H13.
- [ ] Si eso también falla: `git checkout v0-mvp -- qd/` y repetir.
- [ ] `git add entrega/ cache/llm/final/ && git commit && git push`. **Verificar en GitHub** que `entrega/` sea visible en el repo público.
- [ ] Generar `reporte.html` sobre el set final para el pitch (no se commitea en `entrega/`).
- [ ] Cargar en Hacki: link al repo + video. **Al menos 30 minutos de margen antes del cierre.**

### Planes B deterministas (siempre probados antes de H13)

| Falla | Plan B |
|---|---|
| No baja el modelo de embeddings | `--embed tfidf` (TF-IDF de caracteres + SVD 256) |
| API del LLM caída, sin clave o con límite de tasa | `--sin-llm` (léxico + voto mayoritario + TF-IDF para títulos) |
| Error en `ruido.joblib` o `cuentas.joblib` | Reglas de A5 y B6 (`modelos.activo=false`) |
| Louvain se cuelga | Componentes conexas sobre las aristas con w ≥ 2 |
| DuckDB se queda sin memoria en B2 | Partir la consulta por `publicacion_id % 8` y unir los resultados |
| El validador falla por formato | `e09` vuelve a escribir con `fillna("")`, el `strftime` fijo y la unión de IDs |
