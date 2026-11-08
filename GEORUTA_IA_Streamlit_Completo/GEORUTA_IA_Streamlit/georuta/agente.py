"""Q-learning tabular: aprende de recompensas, sin ruta fija programada."""

from dataclasses import dataclass, asdict
import random
import numpy as np
from .entorno import GridWorld, Position, ACTIONS


@dataclass(frozen=True)
class QConfig:
    alpha: float = 0.65
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.93
    max_train_steps: int = 150

    def __post_init__(self) -> None:
        if not (0 < self.alpha <= 1):
            raise ValueError("alpha debe estar en (0, 1].")
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma debe estar en [0, 1].")
        if not (0 <= self.epsilon <= 1 and 0 <= self.epsilon_min <= 1):
            raise ValueError("epsilon y epsilon_min deben estar en [0, 1].")
        if not (0 < self.epsilon_decay <= 1):
            raise ValueError("epsilon_decay debe estar en (0, 1].")
        if self.max_train_steps < 1:
            raise ValueError("max_train_steps debe ser positivo.")

    def to_dict(self) -> dict:
        return asdict(self)


class QLearner:
    def __init__(self, world: GridWorld, config: QConfig | None = None, seed: int = 2026):
        self.world = world
        self.config = config if config is not None else QConfig()
        self.seed = seed
        self.rng = random.Random(seed)
        self.q = np.zeros((world.size * world.size, len(ACTIONS)), dtype=float)
        self.episodes_trained = 0
        self.history: list[dict] = []

    def greedy_action(self, pos: Position, rng: random.Random) -> int:
        values = self.q[self.world.state_index(pos)]
        best = float(np.max(values))
        candidates = [
            i for i, val in enumerate(values)
            if np.isclose(float(val), best, atol=1e-10)
        ]
        return rng.choice(candidates)

    def choose_action(self, pos: Position, exploration: float) -> int:
        if self.rng.random() < exploration:
            return self.rng.randrange(len(ACTIONS))
        return self.greedy_action(pos, self.rng)

    def train(self, episodes: int) -> list[dict]:
        if episodes < 0:
            raise ValueError("El número de partidas no puede ser negativo.")
        created = []
        for _ in range(episodes):
            eps = max(
                self.config.epsilon_min,
                self.config.epsilon * (self.config.epsilon_decay ** self.episodes_trained),
            )
            pos = self.world.start
            total_reward = 0.0
            invalid_count = 0
            reached = False
            steps = 0

            for steps in range(1, self.config.max_train_steps + 1):
                action = self.choose_action(pos, eps)
                next_pos, reward, done, invalid = self.world.transition(pos, action)
                current = self.world.state_index(pos)
                next_state = self.world.state_index(next_pos)
                future = 0.0 if done else float(np.max(self.q[next_state]))
                target = reward + self.config.gamma * future
                self.q[current, action] += (
                    self.config.alpha * (target - self.q[current, action])
                )
                total_reward += reward
                invalid_count += int(invalid)
                pos = next_pos
                if done:
                    reached = True
                    break

            self.episodes_trained += 1
            row = {
                "Partida": self.episodes_trained,
                "Llegó a meta": reached,
                "Movimientos": steps,
                "Recompensa acumulada": round(total_reward, 3),
                "Intentos inválidos": invalid_count,
                "Exploración ε": round(eps, 5),
            }
            self.history.append(row)
            created.append(row)
        return created

    def q_table_rows(self) -> list[dict]:
        rows = []
        for r in range(self.world.size):
            for c in range(self.world.size):
                pos = (r, c)
                if pos in self.world.obstacles:
                    continue
                values = self.q[self.world.state_index(pos)]
                rows.append({
                    "Fila": r, "Columna": c,
                    **{ACTIONS[a][0]: round(float(values[a]), 3)
                       for a in range(len(ACTIONS))},
                })
        return rows
