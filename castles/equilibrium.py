"""Optimal play: approximate the symmetric Nash equilibrium over every legal allocation.

Game 1 is symmetric and constant-sum (15 points per matchup), so an optimal field plays a
mixed equilibrium strategy. Against it, no allocation averages more than 7.5 points.
Any pure allocation can be exploited by some counter (see outputs/counters.csv).

The solver runs regret matching+ in self-play over all 4,598,126 allocations. Each
iteration scores every allocation against the current mixture with the same per-castle
counting trick as the baseline, so no payoff matrix is built. The linearly weighted
average strategy converges to an equilibrium. Its exploitability (best pure response
score minus 7.5) is computed exactly and shows how far from optimal it is.

Run with:  python -m castles.equilibrium [--iterations 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .game import DEFAULT_GAME, Game, all_allocations
from .scoring import baseline_histograms, histograms_from_batches, q_tables, score_keys


def mixture_histograms(game: Game, allocs: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Per-castle soldier-count distributions of a mixed strategy, shape (castles, S + 1)."""
    return np.stack([np.bincount(allocs[:, c], weights=probs, minlength=game.soldiers + 1)
                     for c in range(game.castles)])


def expected_points(game: Game, allocs: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """Expected points of each allocation against a field with these per-castle distributions."""
    hist = hist / hist.sum(axis=1, keepdims=True)
    below = np.zeros_like(hist)
    below[:, 1:] = np.cumsum(hist, axis=1)[:, :-1]
    q = 2 * below + hist
    u = np.zeros(len(allocs))
    for c, value in enumerate(game.values):
        u += value * q[c][allocs[:, c]]
    return u / 2


@dataclass
class EquilibriumResult:
    game: Game
    allocs: np.ndarray
    average: np.ndarray  # probabilities of the averaged strategy
    payoffs: np.ndarray  # each allocation's expected points against `average`
    iterations: int
    history: list[dict] = field(default_factory=list)

    @property
    def exploitability(self) -> float:
        return float(self.payoffs.max() - self.game.total_points / 2)


def solve(game: Game = DEFAULT_GAME, iterations: int = 2000, report_every: int = 50,
          verbose: bool = True, allocs: np.ndarray | None = None) -> EquilibriumResult:
    allocs = all_allocations(game) if allocs is None else allocs
    n = len(allocs)
    regrets = np.zeros(n)
    sigma = np.full(n, 1.0 / n)  # start from the uniform field the baseline assumed
    avg = np.zeros(n)
    avg_hist = np.zeros((game.castles, game.soldiers + 1))
    history = []
    start = time.perf_counter()
    for t in range(1, iterations + 1):
        hist = mixture_histograms(game, allocs, sigma)
        u = expected_points(game, allocs, hist)
        regrets += u - sigma @ u
        np.maximum(regrets, 0, out=regrets)
        avg += t * sigma
        avg_hist += t * hist
        total = regrets.sum()
        sigma = regrets / total if total > 0 else np.full(n, 1.0 / n)
        if t % report_every == 0 or t == iterations:
            best = float(expected_points(game, allocs, avg_hist).max())
            entry = {"iteration": t, "exploitability": best - game.total_points / 2,
                     "seconds": round(time.perf_counter() - start, 1)}
            history.append(entry)
            if verbose:
                print(f"  iter {t:>6}  exploitability {entry['exploitability']:.5f} pts  ({entry['seconds']}s)", flush=True)
    avg /= avg.sum()
    payoffs = expected_points(game, allocs, mixture_histograms(game, allocs, avg))
    return EquilibriumResult(game, allocs, avg, payoffs, iterations, history)


def strong_field_points(game: Game, allocs: np.ndarray, shortlist_path: Path) -> np.ndarray | None:
    """Expected points against the baseline shortlist, as a proxy for strong human players."""
    if not shortlist_path.exists():
        return None
    with shortlist_path.open(newline="") as f:
        rows = [[int(r[f"a{i + 1}"]) for i in range(game.castles)] for r in csv.DictReader(f)]
    hist = histograms_from_batches(game, [np.array(rows, dtype=np.int16)])
    return expected_points(game, allocs, hist.astype(float))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--support-threshold", type=float, default=1e-6,
                   help="export strategies whose equilibrium probability is at least this")
    p.add_argument("--show", type=int, default=15)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    game = DEFAULT_GAME
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Solving for optimal (equilibrium) play over {game.num_allocations:,} allocations...")
    res = solve(game, args.iterations, args.report_every)
    allocs, avg, payoffs = res.allocs, res.average, res.payoffs

    uniform = score_keys(game, q_tables(baseline_histograms(game)), allocs)
    uniform = (uniform - game.total_points) / (2 * (game.num_allocations - 1))
    strong = strong_field_points(game, allocs, out / "shortlist.csv")

    # What to submit: the best responses to the equilibrium field, i.e. allocations that
    # come closest to (or exceed) 7.5 if everyone else plays optimally.
    best = np.lexsort((-avg, -payoffs))[: max(args.show, 50)]
    support = np.flatnonzero(avg >= args.support_threshold)
    support = support[np.argsort(-avg[support], kind="stable")]

    cols = [f"a{i + 1}" for i in range(game.castles)]
    with (out / "equilibrium_support.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*cols, "probability", "points_vs_equilibrium", "points_vs_uniform"])
        for i in support.tolist():
            w.writerow([*allocs[i].tolist(), f"{avg[i]:.9f}", f"{payoffs[i]:.6f}", f"{uniform[i]:.6f}"])
    with (out / "equilibrium_best_responses.csv").open("w", newline="") as f:
        w = csv.writer(f)
        header = [*cols, "points_vs_equilibrium", "equilibrium_probability", "points_vs_uniform"]
        w.writerow(header + (["points_vs_strong_field"] if strong is not None else []))
        for i in best.tolist():
            row = [*allocs[i].tolist(), f"{payoffs[i]:.6f}", f"{avg[i]:.9f}", f"{uniform[i]:.6f}"]
            w.writerow(row + ([f"{strong[i]:.6f}"] if strong is not None else []))
    marg = mixture_histograms(game, allocs, avg)
    with (out / "equilibrium_marginals.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["soldiers", *[f"castle_{v}" for v in game.values]])
        for k in range(game.soldiers + 1):
            w.writerow([k, *[f"{marg[c, k]:.6f}" for c in range(game.castles)]])

    baseline_leader = int(np.argmax(uniform))
    mean_alloc = (marg * np.arange(game.soldiers + 1)).sum(axis=1)
    summary = {
        "method": "regret matching+ self-play over every legal allocation, linear averaging",
        "iterations": res.iterations,
        "exploitability": res.exploitability,
        "meaning": "no pure allocation scores more than 7.5 + exploitability against this field",
        "history": res.history,
        "support_size_at_threshold": int(len(support)),
        "support_threshold": args.support_threshold,
        "probability_mass_in_top_1000": float(np.sort(avg)[::-1][:1000].sum()),
        "expected_soldiers_per_castle": [round(float(x), 2) for x in mean_alloc],
        "baseline_leader": {"allocation": allocs[baseline_leader].tolist(),
                            "points_vs_equilibrium": float(payoffs[baseline_leader])},
        "best_vs_equilibrium": [{"allocation": allocs[i].tolist(), "points_vs_equilibrium": float(payoffs[i]),
                                 "probability": float(avg[i]), "points_vs_uniform": float(uniform[i]),
                                 **({"points_vs_strong_field": float(strong[i])} if strong is not None else {})}
                                for i in best[: args.show].tolist()],
    }
    (out / "equilibrium_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nExploitability {res.exploitability:.5f}: nothing beats 7.5 by more than this against an optimal field.")
    print(f"Equilibrium spreads over {len(support):,} allocations (prob >= {args.support_threshold:g}); "
          f"top 1,000 carry {summary['probability_mass_in_top_1000']:.1%} of the mass.")
    print(f"Average soldiers per castle: {summary['expected_soldiers_per_castle']}")
    bl = summary["baseline_leader"]
    print(f"Baseline leader {tuple(bl['allocation'])} scores {bl['points_vs_equilibrium']:.4f} vs the optimal field.")
    print(f"\nBest submissions against an optimal field:")
    print(f"  {'allocation':<24} {'vs optimal':>10} {'eq prob':>9} {'vs uniform':>10}" + (f" {'vs strong':>9}" if strong is not None else ""))
    for i in best[: args.show].tolist():
        line = f"  {str(tuple(allocs[i].tolist())):<24} {payoffs[i]:>10.4f} {avg[i]:>9.5f} {uniform[i]:>10.4f}"
        print(line + (f" {strong[i]:>9.4f}" if strong is not None else ""))
    print(f"\nWrote equilibrium_support.csv, equilibrium_best_responses.csv, equilibrium_marginals.csv, "
          f"equilibrium_summary.json in {out}.")


if __name__ == "__main__":
    main()
