# Relación con el estado del arte

> Archivo separado del README a propósito, para no pisar las ediciones en curso. Va en `docs/ESTADO_DEL_ARTE.md` y se enlaza desde el README con una línea.

## Qué hay publicado (2021–2026)

**Coordinación.** La forma estándar de operacionalizarla sigue siendo la de CooRnet y CooRTweet: dos cuentas son coordinadas si comparten la misma acción (un link, un post) dentro de una ventana de tiempo τ, con un umbral de repeticiones. Pacheco et al. (ICWSM 2021) generalizaron la idea a redes de rasgos compartidos. Desde entonces el campo se movió en tres direcciones:

- **Modelos nulos.** Ng e Iamnitchi (MISDOOM 2023) extraen el "backbone" de la red de co-acciones contra un modelo nulo con corrección de ruido, en lugar de usar un umbral fijo.
- **Dinámica temporal.** Tardelli et al. (PNAS 2024) definen la coordinación como similitud inesperada y estudian por primera vez cómo evoluciona en el tiempo. Un trabajo de diciembre de 2025 (arXiv 2512.19677) propone un modelo multiplex con decaimiento temporal que supera a los anteriores.
- **Síntesis.** El relevamiento de Mannocci, Mazza, Monreale, Tesconi y Cresci (arXiv 2408.01257, 84 trabajos de 2014 a 2024, revisado en abril de 2026) concluye que la diversidad de métodos y de definiciones dificulta comparar resultados y limita la generalización. Giglietto, Graham y Righetti (2026) suman dos problemas abiertos: la IA generativa y el video corto complican la detección, y sigue sin resolverse cómo distinguir activismo legítimo de manipulación.

**Narrativas.** El esquema dominante es embeddings, reducción con UMAP, HDBSCAN y etiquetado (BERTopic, 2022). Lo más reciente le agrega LLMs: Schneider, Steuber y Dreo Rodosek (arXiv 2605.14354, mayo de 2026) filtran 1,2 millones de posts con un LLM de 122 mil millones de parámetros en modo few-shot, agrupan con UMAP y HDBSCAN, y etiquetan los 41 grupos resultantes con un LLM. Funciona, pero el costo de LLM crece con cada post.

**Postura y origen.** Se tratan casi siempre como tareas separadas: clasificación de stance supervisada o con LLM, y detección de eventos emergentes. Es raro verlas integradas con el agrupamiento y con la coordinación.

## Qué hacemos distinto

| Práctica habitual | Nuestra decisión | Por qué |
|---|---|---|
| Ventana τ con umbral fijo de co-acciones | Lift por par contra un **modelo nulo por permutación** que baraja cuentas dentro de fuente × período × franja de demora | Conserva la actividad y el horario de cada cuenta, así una cuenta hiperactiva no parece coordinada. Es la misma familia de nulos con barajado temporal restringido que se usa en forense de mercados (arXiv 2512.18918). |
| Umbral global o FDR sobre todos los pares | Poda por triángulos, densidad, filtro de hiperactivos y Louvain solo en componentes ralos | El FDR global no funcionó en nuestros datos: los grupos al azar pasaban o las campañas reales se partían. |
| LLM por cada post para filtrar o etiquetar | Embeddings chicos (e5-small) y el **grafo de respuestas**; LLM opcional, una llamada por narrativa | Con esto el costo de LLM depende de cuántas narrativas salen, no de cuántos comentarios entran. Sin API el sistema corre completo. |
| Temas, postura y bots en herramientas separadas | Un pipeline con las cuatro tareas, donde cada una alimenta a la otra | Las respuestas unen narrativas y detectan ruido; la coordinación es un voto para orientar la postura. |
| Postura como clasificación de cada texto | Bipartición espectral del grafo de respuestas por narrativa, más votos para orientar | Una respuesta corta mantiene la postura del padre y una larga suele invertirla. La estructura separa los bandos sin entrenamiento. |
| Generalización rara vez medida | Pseudo-final con días y posts no vistos, más una regla de cambios (mejora en práctica y no empeora en el pseudo-final) | Es nuestra respuesta directa a la crítica del relevamiento sobre la generalización. |

## Qué no resolvemos

- **Orientación de la postura.** Los bandos se separan bien, pero decidir cuál apoya falla en afirmaciones redactadas en negativo. Macro-F1 0,67 en práctica.
- **Coordinación con datos nuevos.** Es la tarea que más cae en el pseudo-final.
- **Evasión.** No probamos rotación agresiva de cuentas ni paráfrasis con IA generativa, que es justamente el problema abierto que marcan Giglietto et al. (2026).
- **Coordinación no es manipulación.** El sistema produce alertas con evidencia para un analista, no veredictos.

## Por qué tiene sentido una hackathon sobre esto

El campo tiene métodos, pero no tiene un banco de prueba común: cada trabajo define coordinación a su manera y se evalúa sobre sus propios datos, que es lo que señala el relevamiento de 2024–2026. Un desafío con verdad oculta, un puntaje automático y una regla de re-ejecución es exactamente esa comparación controlada. El aporte de cada equipo es una pieza medible bajo las mismas reglas.

## Referencias

- Mannocci, L., Mazza, M., Monreale, A., Tesconi, M., Cresci, S. *Detection and Characterization of Coordinated Online Behavior: A Survey.* arXiv:2408.01257 (v2, 2026). https://arxiv.org/abs/2408.01257
- Tardelli, S., Nizzoli, L., Avvenuti, M., Cresci, S., Tesconi, M. *Temporal dynamics of coordinated online behavior: Stability, archetypes, and influence.* PNAS (2024).
- Ng, L. H. X., Iamnitchi, A. *Detecting coordinated behavior via backbone extraction over noise-corrected null models.* MISDOOM 2023. arXiv:2309.12729. https://arxiv.org/abs/2309.12729
- arXiv:2512.19677 (2025). Modelo multiplex con decaimiento temporal para coordinación. https://arxiv.org/abs/2512.19677
- Giglietto, F., Graham, T., Righetti, N. Capítulo sobre comportamiento coordinado e IA generativa (2026).
- Schneider, Steuber, Dreo Rodosek. arXiv:2605.14354 (2026). Narrativas con filtrado LLM, UMAP, HDBSCAN y etiquetado LLM. https://arxiv.org/abs/2605.14354
- arXiv:2512.18918 (2025). Redes forenses con barajado temporal restringido como modelo nulo. https://arxiv.org/abs/2512.18918
- Pacheco, D., et al. *Uncovering Coordinated Networks on Social Media: Methods and Case Studies.* ICWSM 2021.
- Giglietto, F., Righetti, N., Rossi, L., Marino, G. CooRnet (2020) y Righetti, N., Balluff, P. CooRTweet (2023–2025).
- Grootendorst, M. *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794 (2022).
