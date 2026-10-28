"""Phases 1-4: score every legal allocation against the full pool and save a shortlist.

Run with:  python -m castles.baseline [--top-k 10000 | --top-percent P | --min-average X]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .game import DEFAULT_GAME, Game, iter_allocation_batches
from .scoring import (
    average_points,
    baseline_histograms,
    exact_average,
    min_key_for_average,
    q_tables,
    score_keys,
)

POOL_NAME = "all_legal_allocations"


@dataclass
class Selection:
    mode: str  # "top_k" | "top_percent" | "min_average"
    top_k: int | None = None
    top_percent: float | None = None
    min_average: float | None = None

    def effective_k(self, population: int) -> int | None:
        if self.mode == "top_k":
            return self.top_k
        if self.mode == "top_percent":
            return max(1, math.ceil(population * self.top_percent / 100))
        return None


@dataclass
class BaselineResult:
    game: Game
    pool_size: int
    evaluated: int
    allocations: np.ndarray  # shortlist rows, sorted by key desc then allocation asc
    keys: np.ndarray
    ranks: np.ndarray  # competition rank against the full population: 1 + #strictly better
    cutoff_key: int | None
    highest_key: int
    key_sum: int
    timings: dict[str, float]


def _sort_shortlist(allocs: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # lexsort uses the last key as primary: score desc, then allocation asc for display.
    order = np.lexsort(tuple(allocs[:, c] for c in reversed(range(allocs.shape[1]))) + (-keys,))
    return allocs[order], keys[order]


def competition_ranks(sorted_keys_desc: np.ndarray) -> np.ndarray:
    neg = -sorted_keys_desc
    return np.searchsorted(neg, neg, side="left") + 1


def run_baseline(game: Game, selection: Selection) -> BaselineResult:
    timings: dict[str, float] = {}
    pool_size = game.num_allocations
    q = q_tables(baseline_histograms(game))

    # Pass 1: count, sum, and find the cutoff with a bounded top-k buffer.
    t0 = time.perf_counter()
    k = selection.effective_k(pool_size)
    kept = np.empty(0, dtype=np.int64)
    evaluated = 0
    key_sum = 0
    highest = None
    for batch in iter_allocation_batches(game):
        keys = score_keys(game, q, batch)
        evaluated += len(batch)
        key_sum += int(keys.sum())
        batch_max = int(keys.max())
        highest = batch_max if highest is None else max(highest, batch_max)
        if k is not None:
            merged = np.concatenate([kept, keys])
            kept = merged if len(merged) <= k else np.partition(merged, len(merged) - k)[-k:]
    if evaluated != pool_size:
        raise RuntimeError(f"generated {evaluated} allocations, expected {pool_size}")
    if k is not None:
        cutoff = int(kept.min())
    else:
        cutoff = min_key_for_average(game, selection.min_average, pool_size)
    timings["pass1_seconds"] = time.perf_counter() - t0

    # Pass 2: re-enumerate and keep everything at or above the cutoff, so ties survive.
    t0 = time.perf_counter()
    rows, row_keys = [], []
    for batch in iter_allocation_batches(game):
        keys = score_keys(game, q, batch)
        mask = keys >= cutoff
        if mask.any():
            rows.append(batch[mask])
            row_keys.append(keys[mask])
    if rows:
        allocs, keys = _sort_shortlist(np.concatenate(rows), np.concatenate(row_keys))
    else:
        allocs = np.empty((0, game.castles), dtype=np.int16)
        keys = np.empty(0, dtype=np.int64)
    timings["pass2_seconds"] = time.perf_counter() - t0

    # Every strategy scoring above the cutoff is in the shortlist, so ranks are global.
    ranks = competition_ranks(keys)
    return BaselineResult(
        game=game,
        pool_size=pool_size,
        evaluated=evaluated,
        allocations=allocs,
        keys=keys,
        ranks=ranks,
        cutoff_key=cutoff if len(keys) else None,
        highest_key=highest,
        key_sum=key_sum,
        timings=timings,
    )


def castle_columns(game: Game) -> list[str]:
    return [f"a{i + 1}" for i in range(game.castles)]


def write_shortlist(result: BaselineResult, path: Path) -> None:
    game = result.game
    avgs = average_points(game, result.keys, result.pool_size)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "baseline_rank", *castle_columns(game), "baseline_average", "score_key"])
        for i, (alloc, rank, avg, key) in enumerate(
            zip(result.allocations.tolist(), result.ranks.tolist(), avgs.tolist(), result.keys.tolist()), 1
        ):
            w.writerow([i, rank, *alloc, f"{avg:.12f}", key])


def write_full_ranking(game: Game, path: Path) -> None:
    q = q_tables(baseline_histograms(game))
    allocs = np.concatenate(list(iter_allocation_batches(game)))
    allocs, keys = _sort_shortlist(allocs, score_keys(game, q, allocs))
    ranks = competition_ranks(keys)
    avgs = average_points(game, keys, game.num_allocations)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline_rank", *castle_columns(game), "baseline_average", "score_key"])
        chunk = 200_000
        for s in range(0, len(keys), chunk):
            e = s + chunk
            for rank, alloc, avg, key in zip(
                ranks[s:e].tolist(), allocs[s:e].tolist(), avgs[s:e].tolist(), keys[s:e].tolist()
            ):
                w.writerow([rank, *alloc, f"{avg:.12f}", key])


def build_summary(result: BaselineResult, selection: Selection, runtime: float) -> dict:
    game = result.game
    n = result.pool_size
    top = []
    for alloc, rank, key in zip(result.allocations[:10].tolist(), result.ranks[:10].tolist(), result.keys[:10].tolist()):
        top.append({"allocation": alloc, "rank": rank, "average": float(exact_average(game, key, n)), "score_key": key})
    # Sum of (Z - self) over the complete pool equals total_points * N * (N - 1),
    # so the population mean of averages must be total_points / 2.
    population_mean = (result.key_sum - game.total_points * n) / (2 * (n - 1) * n)
    return {
        "game": {
            "name": "Game 1: Head to Head",
            "soldiers": game.soldiers,
            "castle_values": list(game.values),
            "points_per_matchup": game.total_points,
            "tie_rule": "tied castle points are split equally",
            "zero_soldiers_allowed": True,
        },
        "opponent_pool": {
            "name": POOL_NAME,
            "definition": "every legal allocation exactly once, equally weighted; each strategy's own entry excluded",
            "size": n,
            "opponents_per_strategy": n - 1,
        },
        "evaluated": result.evaluated,
        "selection": {
            "mode": selection.mode,
            "top_k": selection.top_k,
            "top_percent": selection.top_percent,
            "min_average": selection.min_average,
            "effective_k": selection.effective_k(n),
            "cutoff_ties_kept": True,
            "tie_display_order": "castle allocation ascending (does not indicate a better strategy)",
        },
        "shortlist_size": int(len(result.keys)),
        "cutoff_score_key": result.cutoff_key,
        "cutoff_average": None if result.cutoff_key is None else float(exact_average(game, result.cutoff_key, n)),
        "highest_score_key": result.highest_key,
        "highest_average": float(exact_average(game, result.highest_key, n)),
        "population_mean_average": population_mean,
        "top_strategies": top,
        "runtime_seconds": round(runtime, 3),
        "timings": {k: round(v, 3) for k, v in result.timings.items()},
    }


def print_summary(summary: dict, out_dir: Path, show: int, result: BaselineResult) -> None:
    sel = summary["selection"]
    print(f"Evaluated {summary['evaluated']:,} allocations against pool '{POOL_NAME}' "
          f"({summary['opponent_pool']['opponents_per_strategy']:,} opponents each).")
    print(f"Population mean average: {summary['population_mean_average']:.6f} (expected {result.game.total_points / 2})")
    crit = {"top_k": f"top_k={sel['top_k']}", "top_percent": f"top {sel['top_percent']}%",
            "min_average": f"min_average={sel['min_average']}"}[sel["mode"]]
    if summary["cutoff_score_key"] is None:
        print(f"Selection {crit}: no strategies qualify.")
    else:
        print(f"Selection {crit}: cutoff average {summary['cutoff_average']:.6f} "
              f"(key {summary['cutoff_score_key']}), shortlist size {summary['shortlist_size']:,} incl. ties.")
    avgs = average_points(result.game, result.keys[:show], result.pool_size)
    print(f"\nTop {min(show, len(result.keys))} strategies:")
    print(f"  {'rank':>5}  {'allocation':<24} {'average':>10}")
    for rank, alloc, avg in zip(result.ranks[:show].tolist(), result.allocations[:show].tolist(), avgs.tolist()):
        print(f"  {rank:>5}  {str(tuple(alloc)):<24} {avg:>10.6f}")
    print(f"\nRuntime {summary['runtime_seconds']}s. Wrote {out_dir / 'shortlist.csv'} and {out_dir / 'summary.json'}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--top-k", type=int, help="keep the best K strategies plus cutoff ties (default 10000)")
    g.add_argument("--top-percent", type=float, help="keep the best P%% of strategies plus cutoff ties")
    g.add_argument("--min-average", type=float, help="keep every strategy averaging at least X points")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--full-ranking", action="store_true", help="also export outputs/full_ranking.csv (large)")
    p.add_argument("--show", type=int, default=15, help="strategies to print in the terminal summary")
    p.add_argument("--soldiers", type=int, default=DEFAULT_GAME.soldiers)
    p.add_argument("--values", type=int, nargs="+", default=list(DEFAULT_GAME.values))
    args = p.parse_args(argv)
    if args.top_k is not None and args.top_k < 1:
        p.error("--top-k must be at least 1")
    if args.top_percent is not None and not 0 < args.top_percent <= 100:
        p.error("--top-percent must be in (0, 100]")
    return args


def selection_from_args(args: argparse.Namespace) -> Selection:
    if args.top_percent is not None:
        return Selection("top_percent", top_percent=args.top_percent)
    if args.min_average is not None:
        return Selection("min_average", min_average=args.min_average)
    return Selection("top_k", top_k=args.top_k or 10_000)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    game = Game(args.soldiers, tuple(args.values))
    selection = selection_from_args(args)
    start = time.perf_counter()
    result = run_baseline(game, selection)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_shortlist(result, out_dir / "shortlist.csv")
    if args.full_ranking:
        write_full_ranking(game, out_dir / "full_ranking.csv")
    summary = build_summary(result, selection, time.perf_counter() - start)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary, out_dir, args.show, result)


if __name__ == "__main__":
    main()
