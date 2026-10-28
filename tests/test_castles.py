"""Validation steps from the README's implementation order."""

import csv
import json
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from castles import analysis, baseline
from castles.game import (
    DEFAULT_GAME,
    Game,
    all_allocations,
    doubled_points_vs,
    head_to_head,
    head_to_head_doubled,
    iter_allocation_batches,
    validate_allocation,
)
from castles.scoring import (
    average_points,
    baseline_histograms,
    exact_average,
    histograms_from_batches,
    min_key_for_average,
    q_tables,
    score_keys,
)

SMALL_GAMES = [Game(10, (1, 2, 3, 4, 5)), Game(7, (3, 1, 2)), Game(12, (1, 2, 3, 4)), Game(0, (1, 2)), Game(5, (2,))]


# 1. Input validation, generation, direct scoring.

@pytest.mark.parametrize("bad", [(100, 0, 0, 0), (50, 50, 0, 0, 1), (101, -1, 0, 0, 0), (99.0, 1, 0, 0, 0), (True, 99, 0, 0, 0)])
def test_validate_rejects_illegal(bad):
    with pytest.raises((ValueError, TypeError)):
        validate_allocation(DEFAULT_GAME, bad)


def test_validate_accepts_zero_castles():
    assert validate_allocation(DEFAULT_GAME, (0, 0, 0, 50, 50)) == (0, 0, 0, 50, 50)


@pytest.mark.parametrize("kwargs", [{"soldiers": -1}, {"values": ()}, {"values": (1, 0)}, {"soldiers": 1.5}])
def test_game_rejects_bad_parameters(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Game(**kwargs)


@pytest.mark.parametrize("game", SMALL_GAMES)
def test_small_generator_matches_brute_force(game):
    expected = sorted(p for p in product(range(game.soldiers + 1), repeat=game.castles) if sum(p) == game.soldiers)
    got = [tuple(r) for r in all_allocations(game).tolist()]
    assert got == expected  # every allocation once, lexicographic order
    assert len(got) == game.num_allocations


def test_order_matters():
    a, b = (0, 0, 0, 50, 50), (50, 50, 0, 0, 0)
    assert head_to_head(DEFAULT_GAME, a, b) == (10.5, 4.5)  # wins 4+5, loses 1+2, ties castle 3
    keys = score_keys(DEFAULT_GAME, q_tables(baseline_histograms(DEFAULT_GAME)), np.array([a, b]))
    assert keys[0] != keys[1]


# 3. Full generator and histogram counts.

def test_full_generator_count_and_uniqueness():
    allocs = all_allocations(DEFAULT_GAME)
    assert len(allocs) == 4_598_126 == DEFAULT_GAME.num_allocations
    assert (allocs >= 0).all() and (allocs.sum(axis=1) == 100).all()
    codes = np.zeros(len(allocs), dtype=np.int64)
    for c in range(5):
        codes = codes * 101 + allocs[:, c]
    assert len(np.unique(codes)) == len(allocs)


def test_baseline_histogram_formula_and_totals():
    hist = baseline_histograms(DEFAULT_GAME)
    assert (hist.sum(axis=1) == 4_598_126).all()
    from math import comb
    assert hist[0].tolist() == [comb(103 - k, 3) for k in range(101)]
    assert (histograms_from_batches(DEFAULT_GAME, iter_allocation_batches(DEFAULT_GAME)) == hist).all()


# 4. Wins, losses, ties, complements, self-match, population mean.

def test_castle_wins_losses_ties():
    g = DEFAULT_GAME
    assert head_to_head(g, (20, 20, 20, 20, 20), (20, 20, 20, 20, 20)) == (7.5, 7.5)
    # Castle 1 lost, castles 2-4 tied (1 + 1.5 + 2 each), castle 5 won.
    assert head_to_head(g, (0, 0, 0, 0, 100), (100, 0, 0, 0, 0)) == (9.5, 5.5)
    # Castles 1-3 tied (0.5 + 1 + 1.5 each), castle 4 won, castle 5 lost.
    assert head_to_head(g, (0, 0, 0, 1, 99), (0, 0, 0, 0, 100)) == (7.0, 8.0)


def test_complements_sum_to_15_on_random_pairs():
    rng = np.random.default_rng(0)
    allocs = all_allocations(Game(30, DEFAULT_GAME.values))
    g = Game(30, DEFAULT_GAME.values)
    for _ in range(500):
        a, b = allocs[rng.integers(len(allocs), size=2)].tolist()
        pa, pb = head_to_head_doubled(g, a, b)
        assert pa + pb == 30
        assert head_to_head(g, a, a) == (7.5, 7.5)


@pytest.mark.parametrize("game", SMALL_GAMES)
def test_population_mean_is_half_points(game):
    allocs = all_allocations(game)
    keys = score_keys(game, q_tables(baseline_histograms(game)), allocs)
    n = len(allocs)
    if n == 1:
        return
    assert sum(exact_average(game, int(k), n) for k in keys) / n == Fraction(game.total_points, 2)


def test_full_population_mean_is_7_5():
    res = baseline.run_baseline(DEFAULT_GAME, baseline.Selection("top_k", top_k=10))
    n = res.pool_size
    assert res.key_sum - 15 * n == 15 * n * (n - 1)


# 2. Count-based scoring vs explicit round robin.

@pytest.mark.parametrize("game", [g for g in SMALL_GAMES if g.num_allocations > 1])
def test_count_method_matches_round_robin(game):
    allocs = all_allocations(game)
    n = len(allocs)
    values = np.array(game.values)
    totals = np.zeros(n, dtype=np.int64)
    for i in range(n):
        per = 2 * (allocs[i] > allocs).astype(np.int64) + (allocs[i] == allocs)
        pts = per @ values
        pts[i] = 0  # exclude self
        totals[i] = pts.sum()
    keys = score_keys(game, q_tables(baseline_histograms(game)), allocs)
    assert (keys - game.total_points == totals).all()
    rr_avg = [Fraction(int(t), 2 * (n - 1)) for t in totals]
    assert [exact_average(game, int(k), n) for k in keys] == rr_avg


def test_custom_histograms_match_round_robin_on_subpool():
    game = Game(10, DEFAULT_GAME.values)
    allocs = all_allocations(game)
    pool = allocs[allocs[:, 4] >= 3]  # castle distributions now differ
    hist = histograms_from_batches(game, [pool])
    keys = score_keys(game, q_tables(hist), allocs)
    for i in range(0, len(allocs), 37):
        direct = sum(head_to_head_doubled(game, allocs[i], b)[0] for b in pool)
        assert keys[i] == direct
    assert (doubled_points_vs(game, pool, allocs[5]) == [head_to_head_doubled(game, b, allocs[5])[0] for b in pool]).all()


# 5. Selection, ties, and exports.

def _full_keys(game):
    allocs = all_allocations(game)
    return allocs, score_keys(game, q_tables(baseline_histograms(game)), allocs)


@pytest.mark.parametrize("k", [1, 5, 17, 100])
def test_top_k_keeps_cutoff_ties(k):
    game = Game(12, DEFAULT_GAME.values)
    allocs, keys = _full_keys(game)
    res = baseline.run_baseline(game, baseline.Selection("top_k", top_k=k))
    cutoff = np.sort(keys)[::-1][k - 1]
    assert res.cutoff_key == cutoff
    assert len(res.keys) == (keys >= cutoff).sum() >= k
    assert (np.diff(res.keys) <= 0).all()
    # Ranks: 1 + number strictly better in the full population.
    for key, rank in zip(res.keys.tolist(), res.ranks.tolist()):
        assert rank == 1 + int((keys > key).sum())
    # Deterministic tie order: allocation ascending within a key.
    pairs = list(zip((-res.keys).tolist(), res.allocations.tolist()))
    assert pairs == sorted(pairs)


def test_percent_and_min_average_modes():
    game = Game(12, DEFAULT_GAME.values)
    allocs, keys = _full_keys(game)
    n = len(keys)
    res = baseline.run_baseline(game, baseline.Selection("top_percent", top_percent=10))
    assert len(res.keys) >= int(np.ceil(n * 0.1))
    threshold = 8.0
    res = baseline.run_baseline(game, baseline.Selection("min_average", min_average=threshold))
    avgs = average_points(game, keys, n)
    assert len(res.keys) == (avgs >= threshold).sum()
    kmin = min_key_for_average(game, threshold, n)
    assert exact_average(game, kmin, n) >= threshold > exact_average(game, kmin - 1, n)
    res = baseline.run_baseline(game, baseline.Selection("min_average", min_average=99))
    assert len(res.keys) == 0 and res.cutoff_key is None


def test_cli_exports_match_score_keys(tmp_path):
    baseline.main(["--soldiers", "15", "--top-k", "50", "--output-dir", str(tmp_path), "--full-ranking"])
    game = Game(15, DEFAULT_GAME.values)
    n = game.num_allocations
    summary = json.loads((tmp_path / "summary.json").read_text())
    with (tmp_path / "shortlist.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert summary["shortlist_size"] == len(rows) >= 50
    assert summary["evaluated"] == n
    assert summary["population_mean_average"] == pytest.approx(7.5)
    q = q_tables(baseline_histograms(game))
    for r in rows:
        alloc = np.array([[int(r[f"a{i}"]) for i in range(1, 6)]])
        key = int(r["score_key"])
        assert score_keys(game, q, alloc)[0] == key
        assert float(r["baseline_average"]) == pytest.approx(float(exact_average(game, key, n)), abs=1e-12)
        assert int(r["score_key"]) >= summary["cutoff_score_key"]
    with (tmp_path / "full_ranking.csv").open() as f:
        assert sum(1 for _ in f) == n + 1


def test_analysis_runs_and_scores_match_direct(tmp_path):
    baseline.main(["--soldiers", "12", "--top-k", "30", "--output-dir", str(tmp_path)])
    analysis.main(["--output-dir", str(tmp_path), "--leaders", "3"])
    game = Game(12, DEFAULT_GAME.values)
    with (tmp_path / "shortlist_reranked.csv").open() as f:
        rows = list(csv.DictReader(f))
    survivors = [tuple(int(r[f"a{i}"]) for i in range(1, 6)) for r in rows]
    for r, s in zip(rows, survivors):
        direct = sum(head_to_head(game, s, o)[0] for o in survivors if o != s) / (len(survivors) - 1)
        assert float(r["shortlist_average"]) == pytest.approx(direct, abs=1e-9)
    for name in ("shortlist_reranked.csv", "counters.csv", "analysis_summary.json"):
        assert (tmp_path / name).exists()
