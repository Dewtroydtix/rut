"""GEORUTA-IA | Aplicación educativa para EXPOMATEMÁTICA JUANPABLINA 2026.

Ejecutar: streamlit run app.py
"""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from georuta import (
    GridWorld, DEFAULT_OBSTACLES, QLearner, QConfig,
    shortest_path, simulate, run_experiments, summarize_results,
)
from georuta.graficos import (
    board_figure, experimental_figure, training_figure, q_heatmap,
)

st.set_page_config(
    page_title="GEORUTA-IA | Aprende a encontrar caminos",
    page_icon="🧭", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    (Path(__file__).parent / "assets" / "estilo.css").read_text(encoding="utf-8"),
    unsafe_allow_html=True,
)

REFERENCE = pd.DataFrame({
    "Partidas": [0, 2, 5, 10, 20, 50],
    "Llegadas": [0, 0, 1, 3, 10, 10],
    "Rutas mínimas": [0, 0, 1, 3, 10, 10],
    "Rutas mínimas (%)": [0, 0, 10, 30, 100, 100],
})
SECTIONS = [
    "🏠 Inicio",
    "🧠 Laboratorio IA",
    "▶️ Simulación",
    "🧪 Experimentos",
    "📊 Resultados",
    "🎓 Aprende y expón",
]


def initialize() -> None:
    if "world" not in st.session_state:
        st.session_state.world = GridWorld()
    if "config" not in st.session_state:
        st.session_state.config = QConfig()
    if "agent" not in st.session_state:
        st.session_state.agent = QLearner(
            st.session_state.world, st.session_state.config, seed=2026
        )
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "results_meta" not in st.session_state:
        st.session_state.results_meta = {}


def reset_agent(reset_experiments: bool = False) -> None:
    st.session_state.agent = QLearner(
        st.session_state.world, st.session_state.config,
        seed=int(st.session_state.get("agent_seed", 2026)),
    )
    st.session_state.last_run = None
    if reset_experiments:
        st.session_state.results = None
        st.session_state.results_meta = {}


def hero(section_name: str, subtitle: str) -> None:
    st.markdown(
        '<div class="hero">'
        '<div class="eyebrow">EXPOMATEMÁTICA JUANPABLINA 2026 · CATEGORÍA C</div>'
        f'<h1>GEORUTA-IA · {section_name}</h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_board(path=None, pos=None, reference=False, title="Tablero geométrico") -> None:
    st.plotly_chart(
        board_figure(
            st.session_state.world, path=path, agent_pos=pos,
            show_reference=reference, title=title,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def navigation_sidebar() -> str:
    world = st.session_state.world
    config = st.session_state.config
    st.sidebar.markdown("## 🧭 GEORUTA-IA")
    st.sidebar.caption("La IA que aprende a encontrar caminos")
    section = st.sidebar.radio(
        "Explorar aplicación", SECTIONS, key="section",
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Tablero actual**")
    min_path = shortest_path(world)
    st.sidebar.caption(
        f"{world.size} × {world.size} · {len(world.obstacles)} obstáculos · "
        f"mínimo: {len(min_path) - 1 if min_path else 'sin ruta'} pasos"
    )

    with st.sidebar.expander("⚙️ Editar obstáculos"):
        st.caption(
            "Selecciona las casillas bloqueadas y aplica el cambio. "
            "Inicio (0, 0) y meta (0, 3) son fijos."
        )
        options = [
            f"({r}, {c})"
            for r in range(world.size) for c in range(world.size)
            if (r, c) not in (world.start, world.goal)
        ]
        selected = st.multiselect(
            "Casillas bloqueadas", options=options,
            default=[f"({r}, {c})" for r, c in sorted(world.obstacles)],
            key="obstacles_editor",
        )
        if st.button("Aplicar nuevo tablero", use_container_width=True):
            obstacles = frozenset(
                tuple(map(int, value.strip("()").split(","))) for value in selected
            )
            proposal = GridWorld(
                size=world.size, start=world.start,
                goal=world.goal, obstacles=obstacles,
            )
            if not shortest_path(proposal):
                st.error("El nuevo tablero no tiene ruta entre inicio y meta.")
            else:
                st.session_state.world = proposal
                reset_agent(reset_experiments=True)
                st.rerun()

    with st.sidebar.expander("🔧 Parámetros de aprendizaje"):
        st.caption(
            "Estos valores no aparecen especificados en el PDF. "
            "Se proponen para esta implementación."
        )
        alpha = st.slider(
            "Tasa de aprendizaje (α)", .05, 1.0,
            float(config.alpha), .05,
        )
        gamma = st.slider(
            "Descuento futuro (γ)", .0, 1.0,
            float(config.gamma), .05,
        )
        epsilon = st.slider(
            "Exploración inicial (ε)", .0, 1.0,
            float(config.epsilon), .05,
        )
        decay = st.slider(
            "Factor de descenso de ε", .50, 1.0,
            float(config.epsilon_decay), .01,
        )
        train_cap = st.number_input(
            "Máximo de pasos por partida de entrenamiento",
            10, 1000, config.max_train_steps, step=10,
        )
        st.number_input(
            "Semilla de la IA", 0, 999_999, 2026,
            key="agent_seed",
            help="Permite repetir entrenamientos con la misma configuración.",
        )
        if st.button("Aplicar y reiniciar IA", use_container_width=True):
            st.session_state.config = QConfig(
                alpha=alpha, gamma=gamma, epsilon=epsilon,
                epsilon_decay=decay,
                max_train_steps=int(train_cap),
            )
            reset_agent(reset_experiments=True)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Python · Streamlit · NumPy · Q-learning real")
    st.sidebar.caption("Sin cuentas de pago ni datos personales.")
    return section


def home() -> None:
    hero("Inicio", "Geometría, movimiento y localización con inteligencia artificial real.")
    left, right = st.columns([1.08, 1], gap="large")
    with left:
        st.subheader("¿Puede una IA aprender a encontrar la ruta más corta?")
        st.write(
            "Este laboratorio enseña cómo un agente virtual explora un laberinto "
            "de casillas, aprende con recompensas y penalizaciones y mejora sus "
            "decisiones sin recibir la ruta correcta ya programada."
        )
        st.markdown(
            '<div class="small-note"><b>Pregunta de investigación:</b> '
            '¿Cómo influye el número de partidas de entrenamiento en la capacidad '
            'de la IA para encontrar una ruta corta dentro de una cuadrícula con obstáculos?'
            '</div>', unsafe_allow_html=True,
        )
        st.write("")
        st.subheader("Objetivos")
        st.markdown(
            "- Representar filas, columnas y desplazamientos.\n"
            "- Entrenar un agente mediante aprendizaje por refuerzo.\n"
            "- Contar pasos y compararlos con la ruta mínima.\n"
            "- Medir porcentajes de éxito en ensayos reproducibles."
        )
        st.info(
            "La referencia matemática se calcula con BFS, pero la IA aprende "
            "con Q-learning. Son procesos diferentes."
        )
    with right:
        show_board(
            reference=True,
            title="Laberinto original y ruta mínima matemática",
        )
    st.subheader("Recorrido del proyecto")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            '<div class="info-card"><b>01 · Entrena</b>'
            '<p>Escoge cuántas partidas practicar y observa las recompensas.</p></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="info-card"><b>02 · Observa</b>'
            '<p>Reproduce movimiento por movimiento la política aprendida.</p></div>',
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            '<div class="info-card"><b>03 · Comprueba</b>'
            '<p>Ejecuta 10 ensayos por grupo y visualiza los porcentajes.</p></div>',
            unsafe_allow_html=True,
        )


def laboratory() -> None:
    hero("Laboratorio IA", "Entrenamiento de un agente que actualiza una tabla Q.")
    world = st.session_state.world
    agent = st.session_state.agent
    minimum = shortest_path(world)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Partidas entrenadas", agent.episodes_trained)
    m2.metric("Casillas", world.size ** 2)
    m3.metric("Obstáculos", len(world.obstacles))
    m4.metric("Mínimo BFS", f"{len(minimum) - 1} pasos" if minimum else "Sin solución")
    st.write("")
    a, b = st.columns([.95, 1.05], gap="large")
    with a:
        st.subheader("Controles de entrenamiento")
        episodes = st.number_input(
            "Partidas adicionales", min_value=0,
            max_value=2_000, value=20, step=1,
            help="Las partidas se SUMAN al entrenamiento que ya tiene la IA.",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧠 Entrenar IA", type="primary", use_container_width=True):
                agent.train(int(episodes))
                st.session_state.last_run = None
                st.success(f"Entrenamiento finalizado: {agent.episodes_trained} partidas acumuladas.")
        with c2:
            if st.button("🔄 Reiniciar IA", use_container_width=True):
                reset_agent()
                st.rerun()
        if st.button("🔍 Probar la ruta aprendida", use_container_width=True):
            st.session_state.last_run = simulate(
                world, st.session_state.agent, max_steps=30, seed=9876
            )
            st.session_state.pop("simulation_frame", None)
        st.caption(
            "Los movimientos se eligen por Q-learning; no se introduce manualmente "
            "la ruta de 9 pasos. La evaluación no modifica la tabla Q."
        )

        last_run = st.session_state.last_run
        if last_run is not None:
            status = "✅ Llegó a la meta" if last_run.reached else "⚠️ No llegó en 30 pasos"
            st.write(f"**Prueba:** {status}")
            z1, z2 = st.columns(2)
            z1.metric("Movimientos usados", last_run.moves)
            z2.metric("¿Ruta mínima?", "Sí" if last_run.optimal else "No")
            st.caption(
                f"Intentos inválidos: {last_run.invalid_count} · "
                f"Recompensa: {last_run.total_reward:.0f}"
            )
        if agent.history:
            st.plotly_chart(
                training_figure(agent.history),
                use_container_width=True,
            )
    with b:
        path = st.session_state.last_run.path if st.session_state.last_run else None
        agent_pos = path[-1] if path else world.start
        show_board(path=path, pos=agent_pos, title="Exploración y recorrido de la IA")

    with st.expander("📘 Ver tabla Q y valores aprendidos"):
        st.caption(
            "Cada número estima la conveniencia de escoger un movimiento "
            "desde una casilla. Cero significa que aún no tiene valor aprendido."
        )
        st.dataframe(
            pd.DataFrame(st.session_state.agent.q_table_rows()),
            use_container_width=True, hide_index=True,
        )
        st.plotly_chart(
            q_heatmap(world, st.session_state.agent),
            use_container_width=True,
        )


def simulation_page() -> None:
    hero("Simulación", "Visualiza paso a paso el comportamiento de la IA entrenada.")
    world = st.session_state.world
    run = st.session_state.last_run
    if run is None:
        st.info(
            "Aún no hay una prueba guardada. Puedes evaluar la IA actual "
            "(aunque tenga 0 partidas) para observar su comportamiento."
        )
        if st.button("Generar prueba de la IA actual", type="primary"):
            st.session_state.last_run = simulate(
                world, st.session_state.agent, max_steps=30, seed=9876
            )
            st.session_state.pop("simulation_frame", None)
            run = st.session_state.last_run
    if run is None:
        show_board(pos=world.start)
        return

    left, right = st.columns([1.1, .9], gap="large")
    with left:
        frame = st.slider(
            "Movimiento a visualizar", 0, run.moves,
            0, key="simulation_frame",
        )
        holder = st.empty()
        holder.plotly_chart(
            board_figure(world, path=run.path[:frame + 1],
                         agent_pos=run.path[frame],
                         title=f"Movimiento {frame} de {run.moves}"),
            use_container_width=True, config={"displayModeBar": False},
        )
        speed = st.slider(
            "Segundos por fotograma", .05, .7, .20, .05,
            key="simulation_speed",
        )
        if st.button("▶️ Reproducir recorrido", type="primary"):
            bar = st.progress(0)
            for i in range(run.moves + 1):
                holder.plotly_chart(
                    board_figure(
                        world, path=run.path[:i + 1],
                        agent_pos=run.path[i],
                        title=f"Movimiento {i} de {run.moves}",
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                bar.progress(i / max(run.moves, 1))
                time.sleep(speed)
            st.caption("Reproducción terminada.")
    with right:
        st.subheader("Resultado de la prueba")
        st.metric("Movimientos", run.moves)
        st.metric("Intentos inválidos", run.invalid_count)
        st.metric("Recompensa acumulada", f"{run.total_reward:g}")
        st.success("Alcanzó la meta.") if run.reached else st.warning(
            "No llegó a la meta dentro del límite de 30 movimientos."
        )
        if run.optimal:
            st.success("¡Encontró la ruta mínima matemática!")
        elif run.reached:
            st.info("Llegó, pero utilizó más pasos que el mínimo.")
        if frame > 0:
            st.markdown(
                f"**Movimiento {frame}:** {run.actions[frame - 1]} · "
                f"recompensa {run.rewards[frame - 1]:g} · "
                f"posición {run.path[frame]}."
            )
            if run.invalid_flags[frame - 1]:
                st.error("El agente intentó un movimiento inválido.")
    with st.expander("Registro completo de movimientos"):
        rows = [
            {
                "Paso": i + 1,
                "Acción": action,
                "Posición final": str(run.path[i + 1]),
                "Recompensa": run.rewards[i],
                "Inválido": "Sí" if run.invalid_flags[i] else "No",
            }
            for i, action in enumerate(run.actions)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def experiments_page() -> None:
    hero("Experimentos", "Compara la IA entrenada con diferentes cantidades de partidas.")
    st.write(
        "Reproducimos el diseño experimental del proyecto: **0, 2, 5, 10, 20 y 50 "
        "partidas**. Para cada grupo se crea una IA nueva y se evalúa con semillas "
        "distintas. El tablero permanece fijo."
    )
    n1, n2, n3 = st.columns(3)
    with n1:
        repeats = st.number_input(
            "Ensayos por cantidad", 1, 50, 10, step=1,
            help="El proyecto original usa 10.",
        )
    with n2:
        seed = st.number_input(
            "Semilla base", 0, 999_999, 2026, step=1,
        )
    with n3:
        st.metric("Máximo por prueba", "30 movimientos")
    if st.button("🧪 Ejecutar experimento completo", type="primary", use_container_width=True):
        with st.spinner("Entrenando y evaluando agentes independientes..."):
            results = run_experiments(
                st.session_state.world, st.session_state.config,
                repeats=int(repeats), base_seed=int(seed),
            )
        st.session_state.results = results
        st.session_state.results_meta = {
            "semilla": int(seed), "ensayos": int(repeats),
            "config": st.session_state.config.to_dict(),
        }
        st.success(
            f"Experimento completado: {len(results)} evaluaciones "
            "y seis cantidades de entrenamiento."
        )
    if st.session_state.results is None:
        st.info("Ejecuta el experimento para generar tus propios datos.")
        st.markdown("**Resultados incluidos en el PDF (solo como referencia):**")
        st.dataframe(REFERENCE, hide_index=True, use_container_width=True)
        return
    summary = summarize_results(st.session_state.results)
    st.plotly_chart(
        experimental_figure(summary, compare_reference=True),
        use_container_width=True,
    )
    st.markdown("**Resumen calculado por el programa**")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.caption(
        "La curva roja representa el documento escolar; las demás son resultados "
        "calculados por esta aplicación. No tienen por qué coincidir."
    )
    st.download_button(
        "⬇️ Descargar todos los ensayos (CSV)",
        st.session_state.results.to_csv(index=False).encode("utf-8-sig"),
        file_name="georuta_ensayos.csv", mime="text/csv",
    )
    st.download_button(
        "⬇️ Descargar resumen (CSV)",
        summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="georuta_resumen.csv", mime="text/csv",
    )


def results_page() -> None:
    hero("Resultados", "Evidencia cuantitativa del entrenamiento y comparación con el proyecto.")
    st.subheader("Datos de referencia documentados")
    st.caption(
        "Estos números proceden de la página 4 del plan escolar. "
        "No son datos generados por esta ejecución."
    )
    st.dataframe(REFERENCE, hide_index=True, use_container_width=True)
    st.markdown(
        "**Cálculo del porcentaje:** "
        "rutas mínimas (%) = (rutas mínimas ÷ número de ensayos) × 100."
    )
    st.write("Ejemplo del documento: 3 rutas mínimas en 10 ensayos = 30 %.")
    st.divider()
    results = st.session_state.results
    if results is None:
        st.info(
            "Aún no se han generado ensayos propios. Abre «Experimentos» "
            "y selecciona «Ejecutar experimento completo»."
        )
        return
    summary = summarize_results(results)
    st.subheader("Nuestros resultados")
    st.plotly_chart(
        experimental_figure(summary, compare_reference=True),
        use_container_width=True,
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)
    meta = st.session_state.results_meta
    st.caption(
        f"Semilla base: {meta.get('semilla', '—')} · "
        f"{meta.get('ensayos', '—')} ensayos por nivel. "
        "Si cambian los hiperparámetros o los obstáculos, hay que repetir el experimento."
    )
    with st.expander("Ver ensayos individuales"):
        st.dataframe(results, hide_index=True, use_container_width=True)
    st.download_button(
        "⬇️ Exportar resultados del experimento",
        results.to_csv(index=False).encode("utf-8-sig"),
        "resultados_georuta.csv", mime="text/csv",
    )


def education_page() -> None:
    hero("Aprende y expón", "Guía didáctica para estudiantes de 5.º y 6.º de primaria.")
    t1, t2, t3 = st.tabs(["🧠 ¿Cómo aprende?", "📐 Matemática", "🎤 Feria escolar"])
    with t1:
        st.subheader("Aprendizaje por refuerzo")
        st.write(
            "Un agente toma una decisión, el entorno responde con una recompensa "
            "y el agente actualiza una tabla Q que guarda cuánto conviene cada acción."
        )
        st.markdown(
            "- **+30**: alcanza la meta.\n"
            "- **−3**: intenta atravesar un muro o salir del tablero.\n"
            "- **−1**: realiza un movimiento válido que todavía no llega a la meta."
        )
        st.latex(
            r"Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]"
        )
        st.caption(
            "Q(s,a): valor aprendido; α: cuánto aprende; γ: importancia "
            "de recompensas futuras. En la meta, el valor futuro es cero."
        )
        st.warning(
            "Los parámetros α, γ y ε son decisiones de esta implementación: "
            "el plan escolar no indica sus valores exactos."
        )
        st.write(
            "Al comenzar, la tabla Q vale cero. Conforme se ejecutan partidas, "
            "el agente utiliza lo aprendido para preferir algunas acciones. "
            "La exploración aleatoria sirve para descubrir caminos nuevos."
        )
    with t2:
        st.subheader("Geometría del tablero")
        st.markdown(
            "**Posición:** (fila, columna).\n\n"
            "**Desplazamientos:** arriba, abajo, izquierda o derecha; sin diagonales.\n\n"
            "**Longitud de ruta:** número de acciones realizadas.\n\n"
            "**Ruta mínima:** la cantidad de acciones de un camino sin desviaciones."
        )
        st.write("Tablero original: 4 filas × 4 columnas = 16 casillas.")
        st.write(
            "Inicio (0, 0), meta (0, 3), obstáculos (0, 1), (1, 1), (2, 1)."
        )
        st.write(
            "En este laberinto: 3 abajo + 3 a la derecha + 3 arriba = **9 movimientos**."
        )
        st.info(
            "La búsqueda en anchura (BFS) calcula una ruta mínima para verificar "
            "el resultado; no forma parte del entrenamiento de la IA."
        )
        show_board(reference=True, title="Demostración matemática del mínimo")
    with t3:
        st.subheader("Guion de exposición de 8 minutos")
        st.dataframe(pd.DataFrame([
            {"Tiempo": "1 minuto", "Actividad": "Título, problema y objetivos"},
            {"Tiempo": "2 minutos", "Actividad": "Geometría y aprendizaje por refuerzo"},
            {"Tiempo": "2 minutos", "Actividad": "Demostración manual y digital"},
            {"Tiempo": "2 minutos", "Actividad": "Tablas, gráficos y comparación"},
            {"Tiempo": "1 minuto", "Actividad": "Conclusiones y preguntas"},
        ]), hide_index=True, use_container_width=True)
        st.subheader("Preguntas que podría hacer el jurado")
        with st.expander("¿Por qué esto sí es inteligencia artificial?"):
            st.write(
                "Porque la tabla Q se modifica mediante experiencia, recompensas y "
                "errores. No contiene una lista fija de los movimientos correctos."
            )
        with st.expander("¿Por qué se necesitan varios ensayos?"):
            st.write(
                "Porque el aprendizaje y los empates entre acciones pueden depender "
                "del azar. Repetir permite calcular porcentajes más informativos."
            )
        with st.expander("¿Por qué 9 es el mínimo?"):
            st.write(
                "Los muros bloquean el paso por la columna 1 hasta la última fila. "
                "Hay que bajar 3, cruzar 3 y subir 3: total 9."
            )
        with st.expander("¿Qué ocurre si cambian los obstáculos?"):
            st.write(
                "El agente debe volver a entrenarse: lo aprendido corresponde "
                "a la distribución de obstáculos anterior."
            )
        st.subheader("Limitaciones")
        st.write(
            "Los resultados dependen de la semilla, las recompensas, el número "
            "de entrenamientos y los hiperparámetros. No se garantiza 100 % "
            "con un número fijo de partidas."
        )
        st.caption(
            "Fuentes: plan GEORUTA-IA, páginas 1–5; "
            "Google Colaboratory; UC Berkeley CS188 Gridworld."
        )


initialize()
section = navigation_sidebar()
if section == SECTIONS[0]:
    home()
elif section == SECTIONS[1]:
    laboratory()
elif section == SECTIONS[2]:
    simulation_page()
elif section == SECTIONS[3]:
    experiments_page()
elif section == SECTIONS[4]:
    results_page()
else:
    education_page()
