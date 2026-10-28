"""Phase 5: re-score the shortlist against itself and other opponent pools, and find counters.

Run after `python -m castles.baseline`:
    python -m castles.analysis [--leaders 10] [--applicants path/to/choices.csv]

Every score keeps the name of the opponent pool it was computed against. Baseline
scores are carried through unchanged so later comparisons retain them.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .game import Game, doubled_points_vs, iter_allocation_batches, validate_allocation
from .scoring import average_points, histograms_from_batches, q_tables, score_keys

Predicate = Callable[[np.ndarray], np.ndarray]


@dataclass
class Pool:
    name: str
    definition: str
    hist: np.ndarray
    size: int
    # Returns how many of the pool's entries are the strategy itself (excluded from its score).
    self_count: Callable[[np.ndarray], np.ndarray]


def _balanced(allocs: np.ndarray) -> np.ndarray:
    return ((allocs >= 10) & (allocs <= 30)).all(axis=1)


def _value_ordered(allocs: np.ndarray) -> np.ndarray:
    return (np.diff(allocs, axis=1) >= 0).all(axis=1)


def _high_value(allocs: np.ndarray) -> np.ndarray:
    return allocs[:, -2:].sum(axis=1) >= 60


FILTER_POOLS: list[tuple[str, str, Predicate]] = [
    ("balanced", "legal allocations with every castle between 10 and 30 soldiers", _balanced),
    ("value_ordered", "legal allocations with soldiers non-decreasing in castle value (a1 <= ... <= a5)", _value_ordered),
    ("high_value_focus", "legal allocations sending at least 60 soldiers to the two most valuable castles", _high_value),
]


def filter_pool(game: Game, name: str, definition: str, predicate: Predicate) -> Pool:
    batches = (b[predicate(b)] for b in iter_allocation_batches(game))
    members = [b for b in batches if len(b)]
    hist = histograms_from_batches(game, members)
    size = sum(len(b) for b in members)
    return Pool(name, definition, hist, size, lambda a: predicate(a).astype(np.int64))


def explicit_pool(game: Game, name: str, definition: str, allocs: np.ndarray, counts: np.ndarray,
                  exclude_self: bool) -> Pool:
    hist = histograms_from_batches(game, [allocs], [counts])
    lookup = {tuple(r): int(c) for r, c in zip(allocs.tolist(), counts.tolist())}

    def self_count(a: np.ndarray) -> np.ndarray:
        if not exclude_self:
            return np.zeros(len(a), dtype=np.int64)
        return np.array([min(lookup.get(tuple(r), 0), 1) for r in a.tolist()], dtype=np.int64)

    return Pool(name, definition, hist, int(counts.sum()), self_count)


def load_applicants(game: Game, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """CSV with columns a1..aN and an optional `count` column (default 1 per row)."""
    rows, counts = [], []
    with path.open(newline="") as f:
        for line in csv.DictReader(f):
            rows.append(validate_allocation(game, [int(line[f"a{i + 1}"]) for i in range(game.castles)]))
            counts.append(int(line.get("count") or 1))
    return np.array(rows, dtype=np.int16), np.array(counts, dtype=np.int64)


def score_against(game: Game, pool: Pool, allocs: np.ndarray) -> np.ndarray:
    keys = score_keys(game, q_tables(pool.hist), allocs)
    return average_points(game, keys, pool.size, pool.self_count(allocs))


def ranks_desc(values: np.ndarray) -> np.ndarray:
    """Competition ranks (1 = best), with exact ties sharing a rank."""
    neg = -np.asarray(values)
    return np.searchsorted(np.sort(neg), neg, side="left") + 1


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def best_responses(game: Game, leader: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
    """Best doubled score any legal allocation gets vs `leader`, how many reach it, and the first one."""
    best, count, example = -1, 0, None
    for batch in iter_allocation_batches(game):
        pts = doubled_points_vs(game, batch, leader)
        m = int(pts.max())
        if m > best:
            best, count, example = m, int((pts == m).sum()), tuple(batch[int(np.argmax(pts))].tolist())
        elif m == best:
            count += int((pts == m).sum())
    return best, count, example


def load_shortlist(path: Path, game: Game) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    cols = [f"a{i + 1}" for i in range(game.castles)]
    return {
        "allocs": np.array([[int(r[c]) for c in cols] for r in rows], dtype=np.int16),
        "baseline_rank": np.array([int(r["baseline_rank"]) for r in rows], dtype=np.int64),
        "baseline_average": np.array([float(r["baseline_average"]) for r in rows]),
        "score_key": np.array([int(r["score_key"]) for r in rows], dtype=np.int64),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--leaders", type=int, default=10, help="leaders to inspect head to head")
    p.add_argument("--applicants", type=Path, help="CSV of actual applicant choices to use as an extra pool")
    p.add_argument("--show", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.perf_counter()
    out: Path = args.output_dir
    base_summary = json.loads((out / "summary.json").read_text())
    game = Game(base_summary["game"]["soldiers"], tuple(base_summary["game"]["castle_values"]))
    sl = load_shortlist(out / "shortlist.csv", game)
    allocs = sl["allocs"]
    if len(allocs) < 2:
        raise SystemExit("shortlist needs at least two strategies; rerun the baseline with a wider cutoff")

    # 1. Survivors vs survivors, using the shortlist's own per-castle histograms.
    pools = [explicit_pool(game, "shortlist", "the baseline shortlist, each survivor once; own entry excluded",
                           allocs, np.ones(len(allocs), dtype=np.int64), exclude_self=True)]
    # 4. Additional opponent pools.
    pools += [filter_pool(game, *spec) for spec in FILTER_POOLS]
    if args.applicants:
        a_allocs, a_counts = load_applicants(game, args.applicants)
        pools.append(explicit_pool(game, "applicants", f"actual applicant choices from {args.applicants}",
                                   a_allocs, a_counts, exclude_self=False))
    # A filter pool can be empty or self-only in small games; it has no opponents to score against.
    skipped = [p.name for p in pools if p.size - int(p.self_count(allocs).max()) < 1]
    pools = [p for p in pools if p.name not in skipped]
    for name in skipped:
        print(f"Skipping pool '{name}': it has no opponents for some survivors in this game.")

    scores = {p.name: score_against(game, p, allocs) for p in pools}
    ranks = {name: ranks_desc(s) for name, s in scores.items()}
    baseline_rank_in_shortlist = ranks_desc(sl["baseline_average"])

    # 2. Compare rankings, keeping both scores and naming each pool.
    with (out / "shortlist_reranked.csv").open("w", newline="") as f:
        w = csv.writer(f)
        header = ["baseline_rank", *[f"a{i + 1}" for i in range(game.castles)], "baseline_average"]
        for p in pools:
            header += [f"{p.name}_average", f"{p.name}_rank"]
        header.append("shortlist_rank_change")
        w.writerow(header)
        order = np.lexsort((sl["baseline_rank"], ranks["shortlist"]))
        for i in order.tolist():
            row = [int(sl["baseline_rank"][i]), *allocs[i].tolist(), f"{sl['baseline_average'][i]:.12f}"]
            for p in pools:
                row += [f"{scores[p.name][i]:.12f}", int(ranks[p.name][i])]
            row.append(int(baseline_rank_in_shortlist[i] - ranks["shortlist"][i]))
            w.writerow(row)

    # 3. Head-to-head: who counters the leaders, inside the shortlist and among all allocations.
    leaders = np.argsort(ranks["shortlist"], kind="stable")[: args.leaders]
    counters = []
    for i in leaders.tolist():
        leader = tuple(allocs[i].tolist())
        vs_leader = doubled_points_vs(game, allocs, leader)
        vs_leader[i] = -1  # skip the self-match
        j = int(np.argmax(vs_leader))
        best, n_best, example = best_responses(game, leader)
        counters.append({
            "leader": leader,
            "leader_shortlist_rank": int(ranks["shortlist"][i]),
            "leader_baseline_rank": int(sl["baseline_rank"][i]),
            "best_shortlist_counter": tuple(allocs[j].tolist()),
            "best_shortlist_counter_points": vs_leader[j] / 2,
            "shortlist_strategies_beating_leader": int((vs_leader > game.total_points).sum()),
            "best_possible_counter": example,
            "best_possible_counter_points": best / 2,
            "allocations_achieving_best_counter": n_best,
        })
    with (out / "counters.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(counters[0]))
        w.writeheader()
        for c in counters:
            w.writerow({k: (" ".join(map(str, v)) if isinstance(v, tuple) else v) for k, v in c.items()})

    def top_of(name: str, n: int = 10) -> list[dict]:
        idx = np.lexsort((sl["baseline_rank"], ranks[name]))[:n]
        return [{"allocation": allocs[i].tolist(), "rank": int(ranks[name][i]), "average": float(scores[name][i]),
                 "baseline_rank": int(sl["baseline_rank"][i])} for i in idx.tolist()]

    summary = {
        "source": {"shortlist": str(out / "shortlist.csv"), "shortlist_size": int(len(allocs)),
                   "baseline_pool": base_summary["opponent_pool"]["name"]},
        "pools": [{"name": p.name, "definition": p.definition, "size": p.size} for p in pools],
        "rank_correlation_vs_baseline": {p.name: spearman(-sl["baseline_average"], -scores[p.name]) for p in pools},
        "top_by_pool": {p.name: top_of(p.name) for p in pools},
        "counters": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()} for c in counters],
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "note": "Scores depend on the opponent pool; no single pool identifies a universal winner.",
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Re-scored {len(allocs):,} shortlisted strategies against {len(pools)} pools:")
    for p in pools:
        print(f"  {p.name:<17} size {p.size:>9,}  Spearman vs baseline {summary['rank_correlation_vs_baseline'][p.name]:+.3f}")
    for p in pools:
        print(f"\nTop {args.show} vs '{p.name}':")
        for t in top_of(p.name, args.show):
            print(f"  {t['rank']:>5}  {str(tuple(t['allocation'])):<24} {t['average']:>9.6f}  (baseline #{t['baseline_rank']})")
    print("\nCounters to the shortlist leaders:")
    for c in counters:
        print(f"  {str(c['leader']):<22} best shortlist counter {str(c['best_shortlist_counter']):<22} "
              f"{c['best_shortlist_counter_points']:>4} pts; best possible {str(c['best_possible_counter']):<22} "
              f"{c['best_possible_counter_points']} pts")
    print(f"\nRuntime {summary['runtime_seconds']}s. Wrote shortlist_reranked.csv, counters.csv, analysis_summary.json in {out}.")


if __name__ == "__main__":
    main()
