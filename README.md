# S-ChessFormer
Repositorio sobre el TFM ''Análisis e interpretabilidad de Transformers aplicados al ajedrez'', de Jorge López Remacho.

## Abstract

Tras la invención y el auge de la arquitectura *Transformer*, su aplicación se ha extendido a un amplio abanico de disciplinas. En 2024, *Google DeepMind* incorporó el ajedrez a esta lista con sus modelos *Searchless Chess*, basados en *Large Scale Transformers* y capaces de alcanzar fuerza de gran maestro en su versión más potente. Este avance abre la puerta al uso de modelos *Transformer* en problemas de planificación y razonamiento a medio y largo plazo.

Este trabajo analiza las capacidades y limitaciones de los modelos *Searchless Chess* a través de tres líneas experimentales. En primer lugar, se evalúan sobre un conjunto de problemas de ajedrez de alta dificultad, diferenciando entre problemas tácticos —que requieren cálculo de secuencias— y estratégicos —donde predominan patrones posicionales y objetivos a largo plazo—. En segundo lugar, se estudia su desempeño en partidas reales contra jugadores humanos y contra distintos niveles del motor Stockfish, mediante bots desplegados en Lichess. Este análisis incluye también partidas de ajedrez *Random-Fisher*, en las que la disposición inicial de las piezas de la primera y última fila se reordena aleatoriamente, lo que permite evaluar la capacidad de generalización de los modelos ante escenarios atípicos. Se analizan métricas como la precisión en partida, la pérdida media de probabilidad de victoria por movimiento y el estilo de juego. Finalmente, se realiza un análisis de interpretabilidad sobre el modelo más pequeño (9 millones de parámetros) mediante técnicas basadas en capas de atención (*Attention Rollout*, *Attention Flow* y diferencias de activaciones), visualización de *embeddings*, métodos basados en gradientes y técnicas agnósticas como SHAP y LIME, adaptadas al contexto del ajedrez.

Los resultados muestran que la principal limitación de estos modelos reside en su falta de profundidad táctica, obteniendo un rendimiento inferior en posiciones que requieren cálculo exhaustivo. En las partidas, se obtiene una caracterización clara del estilo de juego de los modelos, y se observa una degradación en su rendimiento al jugar partidas *Random-Fisher*, aunque en general mantienen un buen nivel de desempeño. Por el contrario, exhiben un comportamiento sólido en contextos estratégicos. Asimismo, se identifican patrones estructurales en las capas de atención que ofrecen pistas sobre los mecanismos internos del modelo, y se constata una notable robustez ante perturbaciones menores en el tablero mediante técnicas como LIME. Estos resultados aportan una caracterización precisa de los límites y fortalezas actuales de los modelos *Transformer* aplicados al ajedrez, y sientan las bases para futuras mejoras en su capacidad de razonamiento, generalización e interpretabilidad.


## Ejecución

Para instalar las dependencias que lanzan el bot, utiliza el siguiente script. Este creará un entorno conda `lichess_bot` con las dependencias necesarias (instalando conda en caso de que no esté presente en el sistema) y creará las variables de entorno precisas para que todo funcione.

```
./install.sh
```

Previo lanzamiento del script asegúrate de dar permisos de ejecución.

```
chmod +x install.sh
```


Para lanzar el bot en lichess, simplemente debes ejecutar los siguientes comandos:

```
cd lichess-bot
```

Antes de lanzar el script principal debes asegurarte de:

- Estás en el entorno conda adecuado. Esto es, en `lichess_bot`. En caso contrario, ejecuta
```
conda activate lichess_bot
```

- Tienes la variable de entorno PYTHONPATH exportada. Si no es así (te saldrá como error que no encuentra el módulo `searchless_chess` al intentar desplegar el bot), ejecuta en el directorio raiz:

```
export "PYTHONPATH=$PYTHONPATH:$(pwd)/lichess_bot"
```

Finalmente, lanza el bot con el siguiente comando:

```
python3 lichess-bot.py --engine_name ENGINE_NAME
```
Donde ENGINE\_NAME puede ser `ThinkLess_9M`, `ThinkLess_136M` o `ThinkLess_270M`.

Alternativamente, si se quiere ejecutar el proceso como un batch en una cola SLURM, se puede emplear el script `run_bot_dios.sh` o `run_bot_dgx.sh`, en función de en qué cola quiera lanzarse. Si se quiere lanzar en otra cola, simplemente se debe indicar al inicio del script.

Si únicamente quieres lanzar un script, por ejemplo, `my_puzzles.py`, debes ejecutar lo siguiente desde el directorio raiz del proyecto:

```
python3 searchless_chess/src/my_puzzles.py --agent <nombre_agente> --input_file problemas/unsolved_puzzles/<archivo_csv> --output_file problemas/solved_puzzles/<archivo_csv>

```
Recuerda instalar Stockfish para poder utilizar el script `my_puzzles`. Esto se puede conseguir con las siguientes instrucciones:


```
cd searchless_chess
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src
make -j profile-build ARCH=x86-64-avx2
cd ../../..
```
