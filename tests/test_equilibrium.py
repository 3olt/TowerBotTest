"""The equilibrium solver against an exact linear-programming solution on small games."""

import numpy as np
import pytest
from scipy.optimize import linprog

from castles.equilibrium import expected_points, mixture_histograms, solve
from castles.game import DEFAULT_GAME, Game, all_allocations, head_to_head


def payoff_matrix(game, allocs):
    values = np.array(game.values)
    rows = []
    for a in allocs:
        per = 2 * (a > allocs).astype(np.int64) + (a == allocs)
        rows.append(per @ values / 2)
    return np.array(rows)


def lp_equilibrium(m):
    # Maximize v subject to x^T M >= v for every column, x a probability vector.
    n = len(m)
    c = np.zeros(n + 1)
    c[-1] = -1
    a_ub = np.hstack([-m.T, np.ones((n, 1))])
    res = linprog(c, A_ub=a_ub, b_ub=np.zeros(n), A_eq=[[1] * n + [0]], b_eq=[1],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    return res.x[:n], res.x[-1]


GAMES = [Game(10, DEFAULT_GAME.values), Game(8, (1, 2, 3)), Game(12, (1, 2, 3, 4))]


@pytest.mark.parametrize("game", GAMES)
def test_expected_points_matches_matrix(game):
    allocs = all_allocations(game)
    m = payoff_matrix(game, allocs)
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(len(allocs)))
    assert np.allclose(expected_points(game, allocs, mixture_histograms(game, allocs, p)), m @ p)
    assert m[0, 1] == head_to_head(game, allocs[0], allocs[1])[0]


@pytest.mark.parametrize("game", GAMES)
def test_solver_approaches_lp_equilibrium(game):
    allocs = all_allocations(game)
    x, value = lp_equilibrium(payoff_matrix(game, allocs))
    assert value == pytest.approx(game.total_points / 2, abs=1e-7)  # symmetric constant-sum game
    exact = expected_points(game, allocs, mixture_histograms(game, allocs, x))
    assert exact.max() == pytest.approx(game.total_points / 2, abs=1e-6)
    res = solve(game, iterations=3000, report_every=3000, verbose=False, allocs=allocs)
    assert res.average.sum() == pytest.approx(1)
    assert 0 <= res.exploitability < 0.02
    assert res.history[-1]["exploitability"] == pytest.approx(res.exploitability)
