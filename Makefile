# Uso:
#   make run                     corre sobre datos/practica -> salida/practica
#   make score                   run + puntaje oficial
#   make desde ETAPA=e04         recalcula desde una etapa (loop en caliente) + puntaje
#   make final                   set final -> entrega/ + validación
#   make humo                    datos de juguete de punta a punta (no necesita el set real)
#   make test                    tests unitarios rápidos
#   make eda                     análisis de la verdad de práctica (H1)
#   make barrido P=narr.q_T V=0.3,0.5,0.7 [ETAPA=e04]
#   make test-sucio              datos ensuciados a propósito (robustez)
#   make instalar                dependencias (uv sync; sin uv, pip install -r requirements.txt)
SET    ?= practica
DATOS  ?= datos/$(SET)
SALIDA ?= salida/$(SET)
PY     ?= $(if $(shell command -v uv 2>/dev/null),uv run python,python)
FLAGS  ?=

.PHONY: run score desde sinllm final validar humo test test-sucio limpiar eda barrido instalar

run:
	$(PY) -m qd.pipeline --datos $(DATOS) --salida $(SALIDA) $(FLAGS)

score: run
	$(PY) evaluacion/puntuar.py --verdad $(DATOS)/verdad --entrega $(SALIDA) --json $(SALIDA)/puntaje.json

desde:
	$(PY) -m qd.pipeline --datos $(DATOS) --salida $(SALIDA) --desde $(ETAPA) $(FLAGS)
	$(PY) evaluacion/puntuar.py --verdad $(DATOS)/verdad --entrega $(SALIDA) --json $(SALIDA)/puntaje.json

sinllm:
	$(MAKE) run FLAGS="--sin-llm $(FLAGS)"

final:
	$(PY) -m qd.pipeline --datos datos/final --salida entrega $(FLAGS)
	$(PY) -m qd.validar --datos datos/final --entrega entrega

validar:
	$(PY) -m qd.validar --datos $(DATOS) --entrega $(SALIDA)

humo:
	$(PY) herramientas/datos_juguete.py --destino work/datos/juguete
	$(MAKE) score SET=juguete DATOS=work/datos/juguete SALIDA=salida/juguete FLAGS="--sin-llm $(FLAGS)"

test-sucio:
	$(PY) herramientas/ensuciar.py --origen $(DATOS) --destino work/datos/sucio
	$(MAKE) run SET=sucio DATOS=work/datos/sucio SALIDA=salida/sucio FLAGS="--sin-llm $(FLAGS)"
	$(PY) -m qd.validar --datos work/datos/sucio --entrega salida/sucio

eda:
	$(PY) herramientas/eda_verdad.py --datos $(DATOS) --salida salida/eda_verdad.md

barrido:
	$(PY) herramientas/barrido.py --datos $(DATOS) --param $(P) --valores $(V) --desde $(or $(ETAPA),e04)

test:
	$(PY) tests/test_basico.py

limpiar:
	rm -rf work/$(SET) $(SALIDA)

instalar:
	@if command -v uv >/dev/null 2>&1; then uv sync; else pip install -r requirements.txt; fi
	@$(PY) -c "from qd.config import cargar, obtener; from sentence_transformers import SentenceTransformer; m = obtener(cargar('config.yaml'), 'embed.modelo', 'intfloat/multilingual-e5-small'); print('make instalar: pre-descargando %s -> caché de Hugging Face (~/.cache/huggingface)' % m); SentenceTransformer(m); print('modelo listo en el caché')"
