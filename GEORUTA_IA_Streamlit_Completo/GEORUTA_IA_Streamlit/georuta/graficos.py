"""Visualizaciones con Plotly para el tablero, las curvas y la tabla Q."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .entorno import GridWorld, Position
from .agente import QLearner
from .matematicas import shortest_path

INK = "#14263D"
GREEN = "#0B826B"
BLUE = "#2463AC"
GOLD = "#ED9E3B"
RED = "#9B2336"
GRID = "#DCE4ED"
BACKGROUND = "#F7F9FC"


def board_figure(
    world: GridWorld,
    path: list[Position] | None = None,
    agent_pos: Position | None = None,
    show_reference: bool = False,
    title: str = "",
) -> go.Figure:
    fig = go.Figure()
    n = world.size
    for row in range(n):
        for col in range(n):
            pos = (row, col)
            if pos in world.obstacles:
                fill = "#334155"
            elif pos == world.start:
                fill = "#C6EEE2"
            elif pos == world.goal:
                fill = "#BDD9FB"
            else:
                fill = "#F1F5F9"
            yy = n - 1 - row
            fig.add_shape(
                type="rect", x0=col - .49, x1=col + .49,
                y0=yy - .49, y1=yy + .49,
                line=dict(color="#FFFFFF", width=2), fillcolor=fill,
            )
            if pos in world.obstacles:
                fig.add_annotation(x=col, y=yy, text="<b>×</b>",
                                   showarrow=False, font=dict(color="white", size=28))
    reference = shortest_path(world) if show_reference else []
    if reference:
        fig.add_trace(go.Scatter(
            x=[c for _, c in reference],
            y=[n - 1 - r for r, _ in reference],
            mode="lines+markers", name="Ruta matemática (BFS)",
            line=dict(color=GREEN, width=4, dash="dot"),
            marker=dict(size=7, color=GREEN), hoverinfo="skip",
        ))
    if path and len(path) > 1:
        fig.add_trace(go.Scatter(
            x=[c for _, c in path],
            y=[n - 1 - r for r, _ in path],
            mode="lines+markers", name="Ruta de la IA",
            line=dict(color=GOLD, width=5),
            marker=dict(size=9, color=GOLD),
            customdata=[[r, c] for r, c in path],
            hovertemplate="Fila %{customdata[0]}, columna %{customdata[1]}<extra></extra>",
        ))
    for pos, label, color in [
        (world.start, "INICIO", GREEN), (world.goal, "META", BLUE),
    ]:
        fig.add_annotation(
            x=pos[1], y=n - 1 - pos[0] + .28,
            text="<b>" + label + "</b>", showarrow=False,
            font=dict(size=10 if n == 4 else 9, color=color),
        )
    if agent_pos is not None:
        fig.add_trace(go.Scatter(
            x=[agent_pos[1]], y=[n - 1 - agent_pos[0]],
            mode="markers+text", text=["IA"], textposition="middle center",
            textfont=dict(size=11, color="white"),
            marker=dict(size=39 if n <= 5 else 29, color=INK,
                        line=dict(color="#FFFFFF", width=3)),
            name="Agente", hovertemplate=f"Agente: {agent_pos}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color=INK)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=550, height=485 if n <= 5 else 550,
        margin=dict(l=38, r=20, b=50, t=43),
        legend=dict(orientation="h", y=-.11, x=.05, font=dict(size=11)),
        xaxis=dict(range=[-.6, n - .4], tickmode="array",
                   tickvals=list(range(n)), title="Columna",
                   showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[-.6, n - .4], tickmode="array",
                   tickvals=list(range(n)), ticktext=list(map(str, range(n - 1, -1, -1))),
                   title="Fila", showgrid=False, zeroline=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        font=dict(family="Arial", color=INK),
    )
    return fig


def experimental_figure(summary: pd.DataFrame, compare_reference: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary["Partidas"], y=summary["Rutas mínimas (%)"],
        name="IA evaluada ahora", mode="lines+markers",
        line=dict(color=GREEN, width=4), marker=dict(size=10),
        hovertemplate="%{x} partidas: %{y:.1f}% mínimo<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=summary["Partidas"], y=summary["Éxito (%)"],
        name="Llegadas a la meta", mode="lines+markers",
        line=dict(color=BLUE, width=2, dash="dot"), marker=dict(size=7),
        hovertemplate="%{x} partidas: %{y:.1f}% llegadas<extra></extra>",
    ))
    if compare_reference:
        fig.add_trace(go.Scatter(
            x=[0, 2, 5, 10, 20, 50], y=[0, 0, 10, 30, 100, 100],
            name="Referencia del documento", mode="lines+markers",
            line=dict(color=RED, width=2, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        ))
    fig.update_layout(
        title="Evolución del aprendizaje: evaluación real",
        xaxis_title="Partidas de entrenamiento",
        yaxis_title="Porcentaje (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BACKGROUND,
        hovermode="x unified", legend=dict(orientation="h", y=-.27),
        margin=dict(l=30, r=20, b=90, t=65), height=380,
    )
    return fig


def training_figure(history: list[dict]) -> go.Figure:
    fig = go.Figure()
    if not history:
        return fig
    fig.add_trace(go.Scatter(
        x=[row["Partida"] for row in history],
        y=[row["Recompensa acumulada"] for row in history],
        mode="lines+markers",
        name="Recompensa por partida", line=dict(color=GREEN, width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        title="Recompensas durante el entrenamiento",
        xaxis_title="Partida", yaxis_title="Recompensa acumulada",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BACKGROUND,
        height=310, margin=dict(l=30, r=10, b=55, t=55),
    )
    return fig


def q_heatmap(world: GridWorld, learner: QLearner) -> go.Figure:
    data = np.full((world.size, world.size), np.nan)
    for row in range(world.size):
        for col in range(world.size):
            if (row, col) not in world.obstacles:
                data[row, col] = float(
                    np.max(learner.q[world.state_index((row, col))])
                )
    fig = go.Figure(go.Heatmap(
        z=data, x=list(range(world.size)), y=list(range(world.size)),
        colorscale=[[0, "#F1F5F9"], [.5, "#B5D7D0"], [1, GREEN]],
        colorbar=dict(title="Valor Q"),
        hovertemplate="Fila %{y}, columna %{x}<br>Valor máximo: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Mejor valor Q por casilla",
        xaxis_title="Columna", yaxis_title="Fila",
        yaxis=dict(autorange="reversed", scaleanchor="x"),
        height=390, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=15, b=45, t=55),
    )
    return fig
