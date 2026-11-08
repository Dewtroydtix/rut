# GEORUTA-IA 🧭

**Laboratorio de inteligencia artificial con Streamlit**  
IEP Juan Pablo Peregrino · EXPOMATEMÁTICA JUANPABLINA 2026 · Categoría C (5.º y 6.º de primaria)

Proyecto basado en el documento escolar *GEORUTA-IA: ¿Puede una inteligencia artificial aprender a encontrar la ruta más corta dentro de un laberinto geométrico?*.

## Requisitos

- Python **3.10 o superior**.
- Una computadora con navegador web.
- Conexión a internet solo para instalar las bibliotecas. Una vez instaladas, la aplicación local no necesita API de pago.

## Inicio rápido

- **Windows:** doble clic en `ejecutar_windows.bat` (instala dependencias e inicia Streamlit).\n- **macOS/Linux:** `bash ejecutar_linux_mac.sh` desde la terminal.\n- También puedes instalar manualmente como se indica a continuación.\n\n## Instalación y ejecución\n
### Windows (PowerShell)

```powershell
cd GEORUTA_IA_Streamlit
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Si PowerShell restringe la activación, se puede usar `.venv\Scripts\python.exe -m pip install -r requirements.txt` y después `.venv\Scripts\python.exe -m streamlit run app.py` sin activar el entorno.

### Linux / macOS

```bash
cd GEORUTA_IA_Streamlit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

La terminal mostrará una dirección local, normalmente `http://localhost:8501`. Abre esa dirección en tu navegador. Para detener el programa presiona `Ctrl+C`.

## ¿Qué incluye?

1. **Inicio:** problema de investigación, hipótesis, objetivos y tablero original.
2. **Laboratorio IA:** entrenamiento incremental de Q-learning, recompensas, tabla Q, mapa de valores y controles para reiniciar el agente.
3. **Simulación:** recorrido paso a paso y animación; indicadores de pasos, errores y recompensa.
4. **Experimentos:** los seis grupos originales (0, 2, 5, 10, 20 y 50 partidas) y 10 ensayos predeterminados por grupo; entrenamiento independiente por ensayo, configurable y reproducible.
5. **Resultados:** comparación entre datos del documento y datos obtenidos en la ejecución; exportación de CSV.
6. **Aprende y expón:** explicación de Q-learning, geometría, guía de presentación de 8 minutos y preguntas del jurado.
7. **Configuración:** edición de los obstáculos (evitando laberintos sin solución), parámetros de aprendizaje y semilla.

## Cómo demostrarlo en una feria

1. Entra a **Inicio** y muestra las 16 casillas y la ruta matemática de **9 movimientos**.
2. Abre **Laboratorio IA** y prueba el agente sin entrenamiento.
3. Entrena el agente (por ejemplo, 5, luego 15 partidas adicionales), prueba su política y observa los cambios. No se garantiza que una cantidad específica de partidas produzca una ruta óptima en todos los casos.
4. Abre **Simulación** para reproducir la ruta aprendida.
5. En **Experimentos**, haz clic en **Ejecutar experimento completo**, observa la tabla y exporta los archivos CSV.
6. En **Resultados**, compara los nuevos porcentajes con la validación incluida en el documento.
7. En **Aprende y expón**, utiliza el guion sugerido para la sustentación.

## Matemática y método experimental

Tablero original:

- 4 filas × 4 columnas = 16 casillas.
- Inicio: `(0, 0)`; meta: `(0, 3)`.
- Muros: `(0, 1)`, `(1, 1)`, `(2, 1)`.
- Movimientos permitidos: arriba, abajo, izquierda y derecha.
- Recompensa meta: `+30`; muro/límite: `−3`; movimiento permitido: `−1`.
- Ruta más corta: 3 abajo + 3 derecha + 3 arriba = **9 movimientos**.
- **BFS** obtiene la ruta mínima solo como referencia matemática, **no entrena la IA**.
- **Q-learning** entrena la IA con prueba, error, recompensas y una tabla Q.
- Una evaluación tiene máximo **30 movimientos**.
- Se compara la tasa de llegadas y la proporción de rutas mínimas.

Para cada combinación de número de partidas y número de ensayo se crea una nueva tabla Q en cero. Se entrena exactamente el número indicado de partidas y luego se hace una evaluación sin modificar Q. Para repetir un experimento se puede reutilizar la misma semilla base. Las semillas varían por ensayo y se emparejan entre los niveles de entrenamiento.

### Parámetros que NO están en el PDF

El PDF no incluye la totalidad de los hiperparámetros ni las semillas originales. Se proponen los siguientes valores para esta implementación, que pueden modificarse desde la barra lateral:

| Parámetro | Valor inicial | Significado |
|---|---:|---|
| α (alpha) | 0.65 | Tasa de aprendizaje |
| γ (gamma) | 0.95 | Factor de descuento |
| ε inicial | 1.00 | Probabilidad de explorar inicialmente |
| ε mínimo | 0.05 | Exploración mínima |
| Descenso de ε | 0.93 | Multiplicador por episodio |
| Pasos máximos durante entrenamiento | 150 | Evita bucles prolongados |
| Pasos máximos durante evaluación | 30 | Según documento |

**Importante:** el registro del documento (0 %, 0 %, 10 %, 30 %, 100 %, 100 %) se presenta solo como referencia. La aplicación **no fuerza** esos valores. Los porcentajes reales se calculan por ejecución y pueden ser distintos.

## Comprobar el funcionamiento

Desde la carpeta del proyecto:

```bash
python -m pytest -q
python -m compileall -q app.py georuta tests
```

Los tests comprueban el camino mínimo de 9 pasos, penalizaciones y recompensas, estado inicial en cero, aprendizaje de Q, reproducibilidad, separación entre evaluación y entrenamiento y resultados de experimentos.

## Estructura

```text
GEORUTA_IA_Streamlit/
├── .streamlit/
│   └── config.toml
├── assets/
│   └── estilo.css
├── georuta/
│   ├── __init__.py
│   ├── entorno.py
│   ├── matematicas.py
│   ├── agente.py
│   ├── evaluacion.py
│   └── graficos.py
├── tests/
│   └── test_georuta.py
├── datos/
│   ├── referencia_documento.csv
│   └── ejemplo_generado_semilla_2026.csv
├── documentacion/
│   └── PLAN_GEORUTA_IA_EXPOMATEMATICA_2026.pdf
├── app.py
├── requirements.txt
├── ejecutar_windows.bat
├── ejecutar_linux_mac.sh
├── GUIA_RAPIDA.txt
└── README.md
```

## Publicación opcional en Streamlit Community Cloud

Sube el contenido de la carpeta a un repositorio GitHub propio y elige `app.py` como archivo de entrada en Streamlit Community Cloud. Los resultados se conservan solo en la sesión actual del navegador, no en una base de datos.

## Privacidad y uso escolar

No se solicitan fotografías, nombres, cuentas privadas ni datos personales. Las pruebas usan semillas y casillas del tablero. El proyecto no utiliza modelos externos, llamadas de pago ni claves API.

## Fuente principal

*IEP Juan Pablo Peregrino. GEORUTA-IA: inteligencia artificial que aprende a encontrar caminos. Plan de proyecto escolar, EXPOMATEMÁTICA JUANPABLINA 2026, páginas 1–5.* 

Las reglas del tablero, recompensas, metodología, resultados de referencia y cronograma proceden de ese documento. La arquitectura Streamlit y los hiperparámetros son decisiones nuevas para esta implementación.

## Archivos de datos incluidos

- `documentacion/PLAN_GEORUTA_IA_EXPOMATEMATICA_2026.pdf`: copia del documento original facilitado para desarrollar la aplicación.
- `datos/referencia_documento.csv`: resultados de referencia de la página 4 (no son mediciones del nuevo software).
- `datos/ejemplo_generado_semilla_2026.csv`: salida de ejemplo calculada por esta implementación utilizando los parámetros predeterminados y una semilla base de 2026. Puedes volver a generar tus propios datos desde la interfaz.

