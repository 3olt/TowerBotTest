# TowerBotTest

Plan for analyzing **Game 1: Head to Head** of the Castle Challenge. The first goal is to evaluate every legal allocation, rank them by average points against all other legal allocations, and save a leading set of strategies for further analysis.

The program below is implemented in the `castles` package. See [Usage](#usage) to run it.

## Usage

Requires Python 3.10+ and NumPy (`pip install -r requirements.txt`).

```bash
python -m castles.baseline                    # phases 1-4: top 10,000 + ties -> outputs/shortlist.csv, outputs/summary.json
python -m castles.baseline --top-percent 0.5  # or a top-percentile cutoff
python -m castles.baseline --min-average 9.8  # or a minimum-average cutoff
python -m castles.baseline --full-ranking     # also export all 4,598,126 strategies (large)
python -m castles.analysis                    # phase 5: shortlist re-ranking, extra pools, counters
python -m castles.analysis --applicants choices.csv   # add real applicant choices (columns a1..a5, optional count)
python -m pytest                              # validation suite (implementation and validation order below)
```

`--soldiers` and `--values` run a smaller or different game, which is how the tests check the exact method against an explicit round robin.

| Module | Contents |
| --- | --- |
| `castles/game.py` | Game parameters, input validation, batched lexicographic allocation generator, direct head-to-head scoring |
| `castles/scoring.py` | Histograms (closed form or from any pool), `q` tables, integer score keys `Z`, exact averages |
| `castles/baseline.py` | Two-pass top-k / percentile / minimum-average selection keeping cutoff ties, CSV/JSON export, terminal summary |
| `castles/analysis.py` | Phase 5: survivors vs survivors, `balanced` / `value_ordered` / `high_value_focus` / applicant pools, rank comparison, counter search |

Phase 5 writes `outputs/shortlist_reranked.csv` (baseline score and rank kept next to each pool's score and rank), `outputs/counters.csv` (for each leader, the best counter inside the shortlist and the best counter among all legal allocations), and `outputs/analysis_summary.json` (pool definitions and sizes, Spearman rank correlation with the baseline, top strategies per pool). Filter pools exclude a strategy's own entry when it belongs to the pool. Applicant pools do not, since a submission really does face every applicant.

## Findings

### Baseline: every legal allocation, equally likely

The full baseline runs in about 0.4 s. All 4,598,126 allocations are scored exactly, and the population mean is exactly 7.5 as required.

| Rank | Allocation (castles 1–5) | Average points |
| --- | --- | --- |
| 1 | `(0, 12, 23, 30, 35)` | 9.8849 |
| 2 | `(0, 11, 23, 30, 36)` | 9.8845 |
| 3 | `(0, 11, 23, 31, 35)` | 9.8842 |
| 4 | `(0, 11, 24, 30, 35)` | 9.8838 |
| 5 | `(0, 12, 22, 30, 36)` | 9.8836 |

- Every top-15 strategy sends 0 soldiers to castle 1, about 11–13 to castle 2, about 23 to castle 3, about 30 to castle 4 and about 35 to castle 5.
- The top 15 lie within 0.003 points of each other. The shape matters more than the exact numbers: give up castle 1 and weight the rest by value.
- The leaders gain about 2.4 points per matchup over the 7.5 population mean.
- The top-10,000 cutoff is 9.7517, with no ties at the cutoff, so the shortlist has exactly 10,000 strategies.

### Shortlist analysis: other opponent pools

| Opponent pool | Leader | Average | Baseline rank |
| --- | --- | --- | --- |
| Shortlist (survivors vs survivors) | `(0, 2, 24, 34, 40)` | 9.93 | #9163 |
| Balanced (each castle 10–30) | `(0, 6, 31, 31, 32)`, tied with several others | 12.00 | #5802 |
| Value-ordered (a1 ≤ … ≤ a5) | `(0, 4, 21, 32, 43)` | 8.46 | #8897 |
| High-value focus (a4 + a5 ≥ 60) | `(1, 14, 19, 24, 42)` | 8.52 | #9966 |

- Rankings depend strongly on the pool. Within the shortlist, rank correlation with the baseline is only 0.09–0.23.
- Against strong opponents (the shortlist itself), the best play nearly gives up castle 2 (about 2 soldiers) and moves the extra soldiers to castles 4 and 5.
- Against balanced opponents, about 31 on each of castles 3, 4 and 5 wins almost every time.
- Every leader can be countered. Some legal allocation takes 12 of 15 points from each shortlist leader, usually by leaving castle 3 empty and stacking castle 5, for example `(1, 3, 0, 35, 61)`. See `outputs/counters.csv`.

### Takeaways

- With no information about opponents, `(0, 12, 23, 30, 35)` is the safest choice.
- Against more strategic opponents, something like `(0, 2–6, 24, 34, 40)` scores better.
- No allocation wins against every pool. The best next step is to score the shortlist against real applicant choices with `python -m castles.analysis --applicants choices.csv`.

## Game rules

Based on the provided challenge screenshot:

- Split exactly 100 soldiers across five labeled castles worth 1, 2, 3, 4, and 5 points.
- Each allocation is a tuple `(a1, a2, a3, a4, a5)` of nonnegative integers whose sum is 100. We assume zero soldiers is allowed.
- In each matchup, the player sending more soldiers to a castle wins its points. A tie splits that castle's points equally.
- A strategy's score is its average **points earned** across matchups. The target is not the percentage of matchups won.

Each matchup awards 15 points in total between the two players. Identical allocations earn 7.5 points each.

## 1. Generate every possible allocation

There are `C(104, 4) = 4,598,126` legal allocations.

Enumerate the first four castle allocations within the remaining soldier budget, then set the fifth to the remainder. This produces every legal allocation exactly once, including strategies that leave some castles empty.

The castles have different values, so changing the order of a tuple produces a different strategy. For example, `(0, 0, 0, 50, 50)` and `(50, 50, 0, 0, 0)` must both be evaluated.

## 2. Calculate exact average scores efficiently

The baseline opponent pool contains every legal allocation once. Score each strategy against every other allocation, excluding its own entry.

A literal round robin would require **10,571,379,056,875 distinct matchups**. Instead, calculate the exact same average using counts of how many opponents send each possible number of soldiers to each castle. This requires no random sampling and no approximation.

For this initial pool, each castle has the same soldier-count distribution. The number of opponent allocations sending exactly `k` soldiers to a particular castle is:

```text
N = 4,598,126
h(k) = C(103 - k, 3), for k = 0, ..., 100
L(t) = sum of h(k) for k < t
q(t) = 2 * L(t) + h(t)
```

Here, `L(t)` counts opponents we beat at that castle, and `h(t)` counts ties. Precompute these 101 values once. For allocation `a`, calculate:

```text
Z(a) = 1*q(a1) + 2*q(a2) + 3*q(a3) + 4*q(a4) + 5*q(a5)
average_points(a) = (Z(a) - 15) / (2 * (N - 1))
```

The subtraction removes the self-match, which contributes 7.5 points, or 15 in the doubled integer calculation. Rank by integer `Z(a)` so equal scores remain exact; convert to decimal points only for display and export.

This works because total points are the sum of the five castle scores. It does not require opponents' allocations at different castles to be independent. Scoring all allocations takes five lookups and a weighted sum per allocation, rather than comparing every pair.

## 3. Filter for the leading strategies

- Start with a configurable `top_k`, initially 10,000, ranked by average points against the full baseline pool.
- Also support a top-percentile or minimum-average cutoff for exploring wider or narrower sets.
- Keep every strategy tied at the cutoff, even if this makes the shortlist larger than `top_k`.
- Preserve each survivor's original average and rank so later comparisons retain the baseline results.
- Use castle allocations as a deterministic display order within exact score ties. This order does not indicate a better strategy.

Generate and score allocations in batches. Keep compact numeric data or a bounded selection structure instead of millions of Python objects. Cutoff ties must be preserved, for example by making a second enumeration pass after finding the cutoff. Do not build a full matchup matrix.

## 4. Save results we can work from

The initial run should produce:

- `outputs/shortlist.csv`: castle allocations, baseline average points, baseline rank, and exact integer score key for every retained strategy.
- `outputs/summary.json`: game parameters, opponent-pool definition, number evaluated, selection settings, actual shortlist size, cutoff score, highest average, and runtime.
- A brief terminal summary showing the highest-scoring strategies and the cutoff used.

Optionally export the complete ranking if needed. The first deliverable is the shortlist and its supporting summary.

## 5. Analyze the shortlist in a later phase

Once the initial filter is available:

1. Recalculate each survivor's average against the other survivors. Build separate soldier-count histograms for each castle and use the same exact counting approach, since the shortlist's castle distributions may differ.
2. Compare these new rankings with the original rankings. Keep both scores and identify the opponent pool used for each.
3. Inspect selected head-to-head matchups to find strategies that counter the leaders.
4. Evaluate promising strategies against additional opponent pools, such as balanced allocations, allocations favoring valuable castles, and actual applicant choices if they become available.

The initial ranking assumes all legal allocations are equally likely. Actual applicants may choose very different patterns, so the best baseline strategy is not automatically the best submission. Filtering can also remove useful counterstrategies. Preserve the original results and allow a wider shortlist instead of repeatedly eliminating strategies and treating the final survivor as a universal winner.

## Implementation and validation order

1. Implement input validation, allocation generation, and a direct two-strategy scoring function.
2. Implement the exact count-based scoring method and compare it against an explicit round robin on a smaller game where every pair is manageable.
3. Check that the full generator yields exactly 4,598,126 unique valid allocations and that the opponent histogram counts sum to the same total.
4. Check castle wins, losses, and half-point ties; complementary matchup scores must sum to 15, and self-matches must score 7.5. Across the complete equally weighted population, the mean of all strategies' averages must also be 7.5.
5. Run the full baseline evaluation, apply the configurable cutoff, and export the shortlist and summary. Confirm that all cutoff ties are included and exported scores match the exact score keys.
6. Use those results to choose the next analysis step together.

Start with Python and exact integer arithmetic, using compact arrays or batching where useful. Profile the complete run before deciding whether further optimization is needed.
