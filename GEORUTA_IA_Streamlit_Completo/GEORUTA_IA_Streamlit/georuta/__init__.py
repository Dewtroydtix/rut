"""GEORUTA-IA: motor de laberintos y aprendizaje por refuerzo."""

from .entorno import GridWorld, ACTIONS, DEFAULT_OBSTACLES
from .agente import QLearner, QConfig
from .matematicas import shortest_path
from .evaluacion import simulate, run_experiments, summarize_results

__all__ = [
    "GridWorld", "ACTIONS", "DEFAULT_OBSTACLES",
    "QLearner", "QConfig", "shortest_path", "simulate",
    "run_experiments", "summarize_results",
]
