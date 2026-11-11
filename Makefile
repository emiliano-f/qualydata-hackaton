# Uso:
#   make run                     corre sobre datos/practica -> salida/practica
#   make score                   run + puntaje oficial
#   make desde ETAPA=e04         recalcula desde una etapa (loop en caliente) + puntaje
#   make final                   set final -> entrega/ + validación
#   make humo                    datos de juguete de punta a punta (no necesita el set real)
#   make test                    tests unitarios rápidos
#   make test-sucio              datos ensuciados a propósito (robustez)
SET    ?= practica
DATOS  ?= datos/$(SET)
SALIDA ?= salida/$(SET)
PY     ?= python
FLAGS  ?=

.PHONY: run score desde sinllm final validar humo test test-sucio limpiar

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

test:
	$(PY) tests/test_basico.py

limpiar:
	rm -rf work/$(SET) $(SALIDA)
