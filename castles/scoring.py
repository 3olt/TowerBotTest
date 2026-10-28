"""Exact count-based scoring against an opponent pool.

For an opponent pool with per-castle soldier histograms H_c(k), a strategy sending
t soldiers to castle c earns, summed over the pool and doubled so ties stay integral,
v_c * (2 * L_c(t) + H_c(t)), where L_c(t) counts opponents sending fewer than t.
Summing over castles gives the integer score key Z. No pairwise comparison is needed
and opponents' allocations at different castles need not be independent.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Iterable

import numpy as np

from .game import Game


def baseline_histograms(game: Game) -> np.ndarray:
    """Closed-form histograms for the pool of every legal allocation once.

    Each castle has h(k) = C(S - k + n - 2, n - 2) opponents sending exactly k soldiers.
    For the default game this is C(103 - k, 3).
    """
    s, n = game.soldiers, game.castles
    if n == 1:
        h = np.zeros(s + 1, dtype=np.int64)
        h[s] = 1
    else:
        h = np.array([comb(s - k + n - 2, n - 2) for k in range(s + 1)], dtype=np.int64)
    return np.tile(h, (n, 1))


def histograms_from_batches(
    game: Game, batches: Iterable[np.ndarray], weights: Iterable[np.ndarray] | None = None
) -> np.ndarray:
    """Per-castle soldier-count histograms of an explicit pool, shape (castles, S + 1)."""
    hist = np.zeros((game.castles, game.soldiers + 1), dtype=np.int64)
    weight_iter = iter(weights) if weights is not None else None
    for batch in batches:
        w = next(weight_iter) if weight_iter is not None else None
        for c in range(game.castles):
            hist[c] += np.bincount(
                batch[:, c], weights=w, minlength=game.soldiers + 1
            ).astype(np.int64)
    return hist


def q_tables(hist: np.ndarray) -> np.ndarray:
    """q_c(t) = 2 * L_c(t) + H_c(t), where L_c(t) = sum of H_c(k) for k < t."""
    below = np.zeros_like(hist)
    below[:, 1:] = np.cumsum(hist, axis=1)[:, :-1]
    return 2 * below + hist


def score_keys(game: Game, q: np.ndarray, allocs: np.ndarray) -> np.ndarray:
    """Integer key Z(a) = sum_c v_c * q_c(a_c): doubled total points vs the whole pool."""
    keys = np.zeros(len(allocs), dtype=np.int64)
    for c, value in enumerate(game.values):
        keys += value * q[c, allocs[:, c]]
    return keys


def average_points(
    game: Game, keys: np.ndarray | int, pool_size: int, self_in_pool: np.ndarray | bool = True
) -> np.ndarray | float:
    """Average points per matchup, excluding the strategy's own pool entry if present.

    The self-match is a full tie worth total_points in doubled units (15 by default).
    """
    self_count = np.asarray(self_in_pool, dtype=np.int64)
    opponents = pool_size - self_count
    return (np.asarray(keys, dtype=np.int64) - game.total_points * self_count) / (2 * opponents)


def exact_average(game: Game, key: int, pool_size: int, self_in_pool: bool = True) -> Fraction:
    s = int(self_in_pool)
    return Fraction(int(key) - game.total_points * s, 2 * (pool_size - s))


def min_key_for_average(game: Game, minimum: float, pool_size: int) -> int:
    """Smallest integer key whose (self-excluded) average is >= `minimum`."""
    target = Fraction(str(minimum)) * 2 * (pool_size - 1) + game.total_points
    return -((-target.numerator) // target.denominator)  # ceil
