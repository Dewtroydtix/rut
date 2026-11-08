"""Pruebas reales: 10 semillas por nivel de entrenamiento, sin valores prefijados."""

from dataclasses import dataclass
import random
import pandas as pd
from .entorno import GridWorld, Position, ACTIONS
from .matematicas import shortest_distance
from .agente import QLearner, QConfig


@dataclass
class Simulation:
    path: list[Position]
    actions: list[str]
    rewards: list[float]
    invalid_flags: list[bool]
    reached: bool
    optimal: bool
    moves: int
    total_reward: float
    min_moves: int | None

    @property
    def invalid_count(self) -> int:
        return sum(self.invalid_flags)


def simulate(
    world: GridWorld, learner: QLearner,
    max_steps: int = 30, seed: int = 2027,
    exploration: float = 0.0,
) -> Simulation:
    """Evaluación aislada: no actualiza la tabla Q."""
    if world != learner.world:
        raise ValueError("El agente y el tablero no coinciden.")
    if max_steps < 1 or not (0 <= exploration <= 1):
        raise ValueError("Los parámetros de evaluación no son válidos.")
    rng = random.Random(seed)
    pos = world.start
    path = [pos]
    actions: list[str] = []
    rewards: list[float] = []
    invalid_flags: list[bool] = []
    reached = False
    for _ in range(max_steps):
        if rng.random() < exploration:
            action = rng.randrange(len(ACTIONS))
        else:
            action = learner.greedy_action(pos, rng)
        next_pos, reward, done, invalid = world.transition(pos, action)
        actions.append(ACTIONS[action][0])
        rewards.append(reward)
        invalid_flags.append(invalid)
        path.append(next_pos)
        pos = next_pos
        if done:
            reached = True
            break

    minimum = shortest_distance(world)
    moves = len(actions)
    return Simulation(
        path=path, actions=actions, rewards=rewards,
        invalid_flags=invalid_flags, reached=reached,
        optimal=bool(reached and minimum is not None and moves == minimum),
        moves=moves, total_reward=sum(rewards), min_moves=minimum,
    )


def run_experiments(
    world: GridWorld,
    config: QConfig | None = None,
    levels: tuple[int, ...] = (0, 2, 5, 10, 20, 50),
    repeats: int = 10,
    base_seed: int = 2026,
    max_eval_steps: int = 30,
) -> pd.DataFrame:
    """Reentrena un agente independiente por combinación (nivel, ensayo).

    Para facilitar la comparación, la misma semilla del ensayo i se reutiliza
    entre niveles; cada entrenamiento comienza con Q en cero.
    """
    if repeats < 1 or any(level < 0 for level in levels):
        raise ValueError("Ensayos y niveles deben ser válidos.")
    rows = []
    config = config if config is not None else QConfig()
    for level in levels:
        for i in range(repeats):
            train_seed = base_seed + i * 1009
            eval_seed = base_seed + 700_000 + i * 1009
            agent = QLearner(world, config, seed=train_seed)
            agent.train(level)
            result = simulate(
                world, agent, max_steps=max_eval_steps, seed=eval_seed
            )
            rows.append({
                "Partidas": level,
                "Ensayo": i + 1,
                "Semilla entrenamiento": train_seed,
                "Semilla evaluación": eval_seed,
                "Llegó": int(result.reached),
                "Ruta mínima": int(result.optimal),
                "Movimientos": result.moves,
                "Movimientos si llegó": result.moves if result.reached else float("nan"),
                "Intentos inválidos": result.invalid_count,
                "Recompensa": result.total_reward,
            })
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    grouped = (
        results.groupby("Partidas", sort=True)
        .agg(
            Ensayos=("Ensayo", "count"),
            Llegadas=("Llegó", "sum"),
            **{
                "Rutas mínimas": ("Ruta mínima", "sum"),
                "Promedio pasos al llegar": ("Movimientos si llegó", "mean"),
                "Promedio intentos inválidos": ("Intentos inválidos", "mean"),
            },
        )
        .reset_index()
    )
    grouped["Éxito (%)"] = (100 * grouped["Llegadas"] / grouped["Ensayos"]).round(1)
    grouped["Rutas mínimas (%)"] = (
        100 * grouped["Rutas mínimas"] / grouped["Ensayos"]
    ).round(1)
    grouped["Promedio pasos al llegar"] = grouped["Promedio pasos al llegar"].round(2)
    grouped["Promedio intentos inválidos"] = (
        grouped["Promedio intentos inválidos"].round(2)
    )
    return grouped
