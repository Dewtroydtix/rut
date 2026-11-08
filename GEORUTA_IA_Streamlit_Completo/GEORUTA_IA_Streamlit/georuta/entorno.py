"""Entorno geométrico discreto, sin diagonales."""

from dataclasses import dataclass, field

Position = tuple[int, int]
ACTIONS: tuple[tuple[str, Position], ...] = (
    ("Arriba", (-1, 0)),
    ("Abajo", (1, 0)),
    ("Izquierda", (0, -1)),
    ("Derecha", (0, 1)),
)
DEFAULT_OBSTACLES = frozenset({(0, 1), (1, 1), (2, 1)})


@dataclass(frozen=True)
class GridWorld:
    size: int = 4
    start: Position = (0, 0)
    goal: Position = (0, 3)
    obstacles: frozenset[Position] = field(default_factory=lambda: DEFAULT_OBSTACLES)
    goal_reward: float = 30.0
    invalid_penalty: float = -3.0
    step_cost: float = -1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacles", frozenset(self.obstacles))
        if not 3 <= self.size <= 10:
            raise ValueError("El tamaño debe estar entre 3 y 10.")
        if not self.inside(self.start) or not self.inside(self.goal):
            raise ValueError("El inicio y la meta deben estar dentro del tablero.")
        if self.start == self.goal:
            raise ValueError("El inicio y la meta deben ser distintos.")
        if any(not self.inside(pos) for pos in self.obstacles):
            raise ValueError("Hay obstáculos fuera del tablero.")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Inicio y meta no pueden ser obstáculos.")

    def inside(self, pos: Position) -> bool:
        row, col = pos
        return 0 <= row < self.size and 0 <= col < self.size

    def passable(self, pos: Position) -> bool:
        return self.inside(pos) and pos not in self.obstacles

    def transition(self, pos: Position, action: int) -> tuple[Position, float, bool, bool]:
        """(siguiente, recompensa, terminó, movimiento inválido)."""
        if not self.passable(pos):
            raise ValueError("La posición de partida no es válida.")
        if pos == self.goal:
            return pos, 0.0, True, False
        if action not in range(len(ACTIONS)):
            raise ValueError("La acción debe estar entre 0 y 3.")
        dr, dc = ACTIONS[action][1]
        new_pos = (pos[0] + dr, pos[1] + dc)
        if not self.passable(new_pos):
            return pos, self.invalid_penalty, False, True
        if new_pos == self.goal:
            return new_pos, self.goal_reward, True, False
        return new_pos, self.step_cost, False, False

    def neighbors(self, pos: Position) -> list[Position]:
        return [
            (pos[0] + dr, pos[1] + dc)
            for _, (dr, dc) in ACTIONS
            if self.passable((pos[0] + dr, pos[1] + dc))
        ]

    def state_index(self, pos: Position) -> int:
        return pos[0] * self.size + pos[1]
