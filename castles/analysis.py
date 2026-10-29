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

from .game import Game, all_allocations, doubled_points_vs, iter_allocation_batches, validate_allocation
from .scoring import average_points, baseline_histograms, histograms_from_batches, q_tables, score_keys

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


def _stack_top_two(allocs: np.ndarray) -> np.ndarray:
    return allocs[:, -2:].sum(axis=1) >= 90


FILTER_POOLS: list[tuple[str, str, Predicate]] = [
    ("balanced", "legal allocations with every castle between 10 and 30 soldiers", _balanced),
    ("value_ordered", "legal allocations with soldiers non-decreasing in castle value (a1 <= ... <= a5)", _value_ordered),
    ("high_value_focus", "legal allocations sending at least 60 soldiers to the two most valuable castles", _high_value),
    ("stack_top_two", "legal allocations sending at least 90 soldiers to the two most valuable castles, e.g. (0, 0, 0, 50, 50)",
     _stack_top_two),
]


def allocation_codes(game: Game, allocs: np.ndarray) -> np.ndarray:
    """Encode each allocation as one integer so membership tests are vectorized."""
    codes = np.zeros(len(allocs), dtype=np.int64)
    for c in range(game.castles):
        codes = codes * (game.soldiers + 1) + allocs[:, c]
    return codes


def filter_pool(game: Game, name: str, definition: str, predicate: Predicate) -> Pool:
    batches = (b[predicate(b)] for b in iter_allocation_batches(game))
    members = [b for b in batches if len(b)]
    hist = histograms_from_batches(game, members)
    size = sum(len(b) for b in members)
    return Pool(name, definition, hist, size, lambda a: predicate(a).astype(np.int64))


def explicit_pool(game: Game, name: str, definition: str, allocs: np.ndarray, counts: np.ndarray,
                  exclude_self: bool) -> Pool:
    hist = histograms_from_batches(game, [allocs], [counts])
    member_codes = allocation_codes(game, allocs[counts > 0])

    def self_count(a: np.ndarray) -> np.ndarray:
        if not exclude_self:
            return np.zeros(len(a), dtype=np.int64)
        return np.isin(allocation_codes(game, a), member_codes).astype(np.int64)

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


def full_search(game: Game, pools: list[Pool], shortlist_allocs: np.ndarray, top_n: int) -> tuple[list[dict], dict]:
    """Score every legal allocation against each pool, so the best reply is not limited to
    the baseline shortlist. Shortlist membership and baseline results are kept for comparison."""
    allocs = all_allocations(game)  # lexicographic, so a stable sort orders exact ties by allocation
    n = game.num_allocations
    keys = score_keys(game, q_tables(baseline_histograms(game)), allocs)
    base_avg = average_points(game, keys, n)
    base_rank = 1 + np.searchsorted(np.sort(-keys), -keys, side="left")
    in_shortlist = np.isin(allocation_codes(game, allocs), allocation_codes(game, shortlist_allocs))

    rows, summary = [], {}
    for pool in pools:
        avg = average_points(game, score_keys(game, q_tables(pool.hist), allocs), pool.size, pool.self_count(allocs))
        order = np.argsort(-avg, kind="stable")
        sorted_neg = -avg[order]
        rank_of = lambda i: int(np.searchsorted(sorted_neg, -avg[i], side="left")) + 1
        for i in order[:top_n].tolist():
            rows.append({"pool": pool.name, "pool_rank": rank_of(i), **{f"a{c + 1}": int(x) for c, x in enumerate(allocs[i])},
                         "pool_average": f"{avg[i]:.12f}", "baseline_rank": int(base_rank[i]),
                         "baseline_average": f"{base_avg[i]:.12f}", "in_shortlist": bool(in_shortlist[i])})
        best = int(order[0])
        sl_best = int(order[np.flatnonzero(in_shortlist[order])[0]])
        summary[pool.name] = {
            "best_allocation": allocs[best].tolist(), "best_average": float(avg[best]),
            "best_baseline_rank": int(base_rank[best]), "best_in_shortlist": bool(in_shortlist[best]),
            "best_shortlist_allocation": allocs[sl_best].tolist(), "best_shortlist_average": float(avg[sl_best]),
            "best_shortlist_rank_among_all": rank_of(sl_best),
            "shortlist_gap": float(avg[best] - avg[sl_best]),
            f"top_{top_n}_in_shortlist": int(in_shortlist[order[:top_n]].sum()),
        }
    return rows, summary


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
    p.add_argument("--search-top", type=int, default=100,
                   help="best allocations per pool to export from the search over all allocations")
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
    # A filter pool can be empty or a single allocation in small games, leaving its member no opponents.
    skipped = [p.name for p in pools if p.size < 2]
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

    # 4b. Search every legal allocation against each pool, keeping the shortlist for comparison.
    search_rows, search_summary = full_search(game, pools, allocs, args.search_top)
    with (out / "pool_search.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(search_rows[0]))
        w.writeheader()
        w.writerows(search_rows)

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
        "full_search": search_summary,
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
    print("\nBest allocation against each pool, searching all allocations (vs best inside the shortlist):")
    for name, s in search_summary.items():
        print(f"  {name:<17} {str(tuple(s['best_allocation'])):<22} {s['best_average']:>8.4f} (baseline #{s['best_baseline_rank']:,})"
              f" | shortlist best {str(tuple(s['best_shortlist_allocation'])):<22} {s['best_shortlist_average']:>8.4f}"
              f" (#{s['best_shortlist_rank_among_all']:,} overall)")
    print(f"\nRuntime {summary['runtime_seconds']}s. Wrote shortlist_reranked.csv, counters.csv, pool_search.csv, "
          f"analysis_summary.json in {out}.")


if __name__ == "__main__":
    main()
