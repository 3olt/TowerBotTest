"""Game definition, input validation, allocation generation, and direct scoring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Iterator, Sequence

import numpy as np

ALLOC_DTYPE = np.int16


@dataclass(frozen=True)
class Game:
    """Split `soldiers` across castles worth `values` points each."""

    soldiers: int = 100
    values: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self) -> None:
        if isinstance(self.soldiers, bool) or not isinstance(self.soldiers, int):
            raise TypeError("soldiers must be an integer")
        if self.soldiers < 0:
            raise ValueError("soldiers must be nonnegative")
        if self.soldiers > np.iinfo(ALLOC_DTYPE).max:
            raise ValueError("soldiers is too large for the allocation dtype")
        values = tuple(self.values)
        if not values:
            raise ValueError("at least one castle is required")
        for v in values:
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError("castle values must be positive integers")
        object.__setattr__(self, "values", values)

    @property
    def castles(self) -> int:
        return len(self.values)

    @property
    def total_points(self) -> int:
        """Points awarded per matchup; also the doubled score of a self-match."""
        return sum(self.values)

    @property
    def num_allocations(self) -> int:
        return comb(self.soldiers + self.castles - 1, self.castles - 1)


DEFAULT_GAME = Game()


def validate_allocation(game: Game, allocation: Sequence[int]) -> tuple[int, ...]:
    """Return `allocation` as a tuple, raising if it is not legal for `game`."""
    alloc = tuple(allocation)
    if len(alloc) != game.castles:
        raise ValueError(f"expected {game.castles} castles, got {len(alloc)}")
    for x in alloc:
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
            raise TypeError(f"soldier counts must be integers, got {x!r}")
        if x < 0:
            raise ValueError(f"soldier counts must be nonnegative, got {x}")
    if sum(alloc) != game.soldiers:
        raise ValueError(f"allocation must sum to {game.soldiers}, got {sum(alloc)}")
    return tuple(int(x) for x in alloc)


@lru_cache(maxsize=None)
def _small_compositions(total: int, parts: int) -> np.ndarray:
    return _build_compositions(total, parts)


def _build_compositions(total: int, parts: int) -> np.ndarray:
    if parts == 1:
        return np.array([[total]], dtype=ALLOC_DTYPE)
    blocks = []
    for first in range(total + 1):
        rest = compositions(total - first, parts - 1)
        block = np.empty((len(rest), parts), dtype=ALLOC_DTYPE)
        block[:, 0] = first
        block[:, 1:] = rest
        blocks.append(block)
    return np.concatenate(blocks)


def compositions(total: int, parts: int) -> np.ndarray:
    """All nonnegative integer `parts`-tuples summing to `total`, in lexicographic order."""
    # Only cache the small inner tables; the full outer table would be large.
    if parts <= 3:
        return _small_compositions(total, parts)
    return _build_compositions(total, parts)


def iter_allocation_batches(game: Game = DEFAULT_GAME) -> Iterator[np.ndarray]:
    """Yield every legal allocation exactly once, batched by the first castle.

    Batches are arrays of shape (rows, castles) in lexicographic order.
    """
    if game.castles == 1:
        yield np.array([[game.soldiers]], dtype=ALLOC_DTYPE)
        return
    for first in range(game.soldiers + 1):
        rest = compositions(game.soldiers - first, game.castles - 1)
        batch = np.empty((len(rest), game.castles), dtype=ALLOC_DTYPE)
        batch[:, 0] = first
        batch[:, 1:] = rest
        yield batch


def all_allocations(game: Game = DEFAULT_GAME) -> np.ndarray:
    return np.concatenate(list(iter_allocation_batches(game)))


def head_to_head_doubled(game: Game, a: Sequence[int], b: Sequence[int]) -> tuple[int, int]:
    """Direct matchup score, doubled so ties stay integral. Sums to 2 * total_points."""
    a = validate_allocation(game, a)
    b = validate_allocation(game, b)
    pa = pb = 0
    for value, x, y in zip(game.values, a, b):
        if x > y:
            pa += 2 * value
        elif y > x:
            pb += 2 * value
        else:
            pa += value
            pb += value
    return pa, pb


def head_to_head(game: Game, a: Sequence[int], b: Sequence[int]) -> tuple[float, float]:
    pa, pb = head_to_head_doubled(game, a, b)
    return pa / 2, pb / 2


def doubled_points_vs(game: Game, allocs: np.ndarray, opponent: Sequence[int]) -> np.ndarray:
    """Vectorized doubled points each row of `allocs` earns against one opponent."""
    opp = np.asarray(validate_allocation(game, opponent), dtype=ALLOC_DTYPE)
    values = np.asarray(game.values, dtype=np.int64)
    per_castle = 2 * (allocs > opp).astype(np.int64) + (allocs == opp)
    return per_castle @ values
