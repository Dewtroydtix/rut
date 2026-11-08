"""Pruebas de geometría, aprendizaje, reproducibilidad y experimentación."""

import numpy as np
import pandas as pd
import pytest

from georuta import (
    GridWorld, DEFAULT_OBSTACLES, QConfig, QLearner,
    shortest_path, simulate, run_experiments, summarize_results,
)


def test_tablero_original_y_minimo():
    w = GridWorld()
    assert w.size == 4
    assert w.obstacles == DEFAULT_OBSTACLES
    assert len(shortest_path(w)) - 1 == 9
    assert shortest_path(w)[0] == (0, 0)
    assert shortest_path(w)[-1] == (0, 3)


def test_transiciones_recompensas():
    w = GridWorld()
    assert w.transition((0, 0), 3) == ((0, 0), -3.0, False, True)
    assert w.transition((0, 0), 1) == ((1, 0), -1.0, False, False)
    assert w.transition((1, 3), 0) == ((0, 3), 30.0, True, False)
    assert w.transition((0, 0), 0) == ((0, 0), -3.0, False, True)


def test_invalidez_y_tablero_sin_ruta():
    with pytest.raises(ValueError):
        GridWorld(obstacles={(0, 0)})
    with pytest.raises(ValueError):
        GridWorld(size=2)
    w = GridWorld(obstacles={(0, 1), (1, 0)})
    assert shortest_path(w) == []


def test_cero_partidas_no_actualiza_tabla():
    w = GridWorld()
    agent = QLearner(w)
    assert agent.train(0) == []
    assert agent.episodes_trained == 0
    assert np.count_nonzero(agent.q) == 0


def test_entrenamiento_modifica_q_y_evaluacion_no_aprende():
    w = GridWorld()
    agent = QLearner(w, seed=22)
    rows = agent.train(30)
    assert len(rows) == 30
    assert np.count_nonzero(agent.q) > 0
    q_before = agent.q.copy()
    result = simulate(w, agent, seed=7)
    assert np.array_equal(q_before, agent.q)
    assert len(result.path) == result.moves + 1
    assert result.moves <= 30


def test_reproducibilidad():
    w = GridWorld()
    a1 = QLearner(w, seed=17)
    a2 = QLearner(w, seed=17)
    assert a1.train(25) == a2.train(25)
    assert np.array_equal(a1.q, a2.q)
    assert simulate(w, a1, seed=99) == simulate(w, a2, seed=99)


def test_experimentos_repetidos_son_iguales_y_no_alteran_agente():
    w = GridWorld()
    config = QConfig()
    first = run_experiments(w, config, repeats=3, base_seed=33)
    second = run_experiments(w, config, repeats=3, base_seed=33)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 18
    assert sorted(first["Partidas"].unique().tolist()) == [0, 2, 5, 10, 20, 50]
    summary = summarize_results(first)
    assert (summary["Ensayos"] == 3).all()
    assert summary["Rutas mínimas"].le(summary["Llegadas"]).all()
    assert summary["Rutas mínimas (%)"].between(0, 100).all()


def test_agente_puede_aprender_ruta_minima_con_entrenamiento_suficiente():
    w = GridWorld()
    agent = QLearner(w, seed=55)
    agent.train(500)
    outcomes = [simulate(w, agent, seed=s) for s in range(10)]
    assert all(outcome.reached for outcome in outcomes)
    assert all(outcome.optimal for outcome in outcomes)


def test_errores_de_parametros():
    with pytest.raises(ValueError):
        QConfig(alpha=0)
    with pytest.raises(ValueError):
        QConfig(gamma=1.1)
    with pytest.raises(ValueError):
        QLearner(GridWorld()).train(-1)
    with pytest.raises(ValueError):
        simulate(GridWorld(), QLearner(GridWorld()), max_steps=0)
