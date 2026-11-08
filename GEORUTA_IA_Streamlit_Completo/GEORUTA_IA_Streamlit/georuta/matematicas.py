"""Ruta mínima por búsqueda en anchura (BFS): referencia matemática, NO IA."""

from collections import deque
from .entorno import GridWorld, Position


def shortest_path(world: GridWorld) -> list[Position]:
    """Retorna posiciones (incluye inicio/meta), o [] si no hay solución."""
    queue = deque([world.start])
    parents: dict[Position, Position | None] = {world.start: None}
    while queue:
        current = queue.popleft()
        if current == world.goal:
            path: list[Position] = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return list(reversed(path))
        for candidate in world.neighbors(current):
            if candidate not in parents:
                parents[candidate] = current
                queue.append(candidate)
    return []


def shortest_distance(world: GridWorld) -> int | None:
    path = shortest_path(world)
    return len(path) - 1 if path else None
