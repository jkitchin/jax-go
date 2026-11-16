# Root-stage time on the minlp.org stage-2b models (2026-11-16)

**Question.** Why do the root stages leave so little of a 60 s cold run to the search, and what can be cut without weakening any bound?

**Short answer.** Two kinds of cost, neither of them compile:

- **`bound_domain`** took 8–15 s cold on the 12 models with declared-free variables, mostly eager JAX sweeps.
- **The root's optional extras** took 15–22 s after the root bound: polish, the OA wait, root local solves, probing and dives. Their budgets either grew with the elapsed time, which cold is mostly compile, or had no limit.

Both are now cut. The search starts about 11 s earlier on average, and no bound is affected.

## 1. Diagnosis

**Method.** `bnb._TRACE` now records the incumbent and the root bound at each stage, plus sub-stage events for polish, OA and probing. A harness also counts the local solves in each stage. Each run uses a cold cache and a 60 s limit.

### `bound_domain`

| model | free variables | `bound_domain` | `monotone_bounds` | `used_variables` | FBBT kernel wait |
|---|---:|---:|---:|---:|---:|
| m394 | 20 | 11.2 s | 6.2 s | 3.7 s | about 2.5 s |
| m474 | 1 | 13.3 s | 7.6 s | 5.0 s | |

- In `monotone_bounds`, the eager FBBT sweep for the row upper bounds took 3.8–4.6 s and the eager Jacobian 2.2–2.5 s.
- Eager JAX compiles one XLA program per primitive and shape: 227–239 compiles of about 30 ms each.
- `used_variables` was called 3 times.

### Root stages after the root bound (base, load 2–6)

| stage | cost | what it gained |
|---|---|---|
| polish of the first root incumbent | m394 5.6 s, m399 4.8 s, m472 5.1 s, m474 4.9 s | nothing in every case |
| OA wait at the root | 4.5 s on every model with binaries | nothing: the cut program never compiled in time cold |
| root local solves | m472: 4 batches, 32 starts, 10.9 s, giving ub ≈ 0 against an optimum of 361; m474: 5.8 s | the search's node-box solves found better |
| probing | m400: 10 s | no bound change |
| root dives | m426: 20 s; CLay: 4–5 s | CLay incumbents 24684–55529 against an optimum of 8092.5 |

### `obbt="tree"`

One OBBT kernel call of 16 boxes (2 LPs × 80 variables × 2 rounds per box) took 131 s on m6. The deadline is checked only before a call starts.

## 2. Fix

### NumPy structure sweeps (`domain.py`)

- **Scalar program.** The joint tape is scalarized once per `bound_domain` call (`_scalar_program`).
- **Dependence.** It comes from `_row_structure`, so `used_variables` and `objective_only_bounds` need no JAX.
- **Affine coefficients.** `_linear_coefs` propagates exact coefficients through add, sub, neg, multiplication by a constant and division by a constant. A variable reached through any other operation gets NaN, which matches `_row_structure`'s `nl`.
- **Row upper bounds.** `_interval_rows` is a NumPy interval forward sweep. 0·∞ is taken as 0, and each operation is widened outward by 2⁻⁵⁰ relative. The old eager sweep was not rounded outward at all.
- **Fallback.** An unhandled operation returns None, and the old JAX path runs (switch: `_NUMPY_STRUCTURE`).
- **Soundness.** The caps only need valid row upper bounds. They are the same argument as before, so the box still keeps an optimal point.

### Root budgets (`bnb.py`, finite time limit only)

| setting | value | limits |
|---|---:|---|
| `_ROOT_LOCAL_SHARE` | 0.15 of the limit | all root-stage local solves together |
| `_POLISH_ROOT_SHARE` | 0.05 of the limit | root polish in all |
| `_OA_ROOT_WAIT` | 0 | the root's wait for the OA cut program |
| `_PROBE_SHARE` | 0.1 of the time left | probing; the most fractional binaries first |
| `_ROOT_DIVE_SHARE` | 0.1 of the limit | the root dive's kernel chunks |
| `_TREE_OBBT_SHARE` | 0.5 of the time left | each batch's OBBT with `obbt="tree"` |

- Local solves, polish, dives and OA only propose incumbents, and every candidate is still screened.
- Probing and OBBT only tighten, so doing less of them is sound.
- Candidate screens use two batch sizes (8 and 64), both compiled in the background.

## 3. Measurements

### Stage traces, base then new per model (same-load pairs)

| model | load | `bound_domain` base → new | root after bound base → new | search starts base → new | result base → new |
|---|---|---|---|---|---|
| m474 | 6 | 11.8 → 3.3 | 16.2 → 9.9 | 51.8 → 34.2 | TL 6146.6 → **optimal 55.6 s** |
| m399 | 5 | 9.7 → 3.3 | 18.4 → 10.5 | 43.3 → 30.6 | TL 28351 → **optimal 57.0 s** |
| m406 | 5 | 9.4 → 3.3 | 16.0 → 8.9 | 41.3 → 28.5 | TL → **optimal 54.4 s** |
| m394 | 6 | 9.2 → 2.8 | 16.1 → 11.1 | 41.6 → 32.0 | TL 8218.9 / 5080 → TL 8092.5 / 7227 |
| m391 | 6 | 8.6 → 3.2 | 15.9 → 8.9 | 35.8 → 29.8 | TL 8275 / 7912 → TL 8092.5 / 7920 |
| m396 | 8–11 | 9.5 → 3.4 | 16.3 → 8.7 | 44.8 → 32.7 | optimal 56.5 → optimal 54.2 |
| m403 | 11–14 | 10.5 → 4.2 | 19.7 → 8.8 | 53.0 → 36.1 | TL → **optimal 53.0 s** |
| m420 | 12 | — | 19.6 → 9.4 | 43.9 → 35.7 | bound 0.080 → 0.122 |
| m421 | 15 | — | 20.5 → 11.4 | 49.3 → 39.7 | bound 0.427 → 0.584 |
| m422 | 15–17 | — | 21.8 → 9.5 | 53.6 → 41.6 | bound 0 → 1.128; nodes 127 → 1535 |
| m426 | 13–16 | — | 26.7 → 8.6 | 54.1 → 28.7 | nodes 63 → 511 |

- Means over the 13 runs with a root bound: `bound_domain` 11.8 → 5.1 s, root stages after the root bound 15.1 → 8.3 s, search start 48.5 → 37.0 s.
- m410, m34, m400, m441 and m47 got no root bound in either arm at load 13–19: compile time.
- In earlier single runs at load about 3, the new code solved m474 (55.9 s), m399 (55.2 s), m394 (57.6 s) and m406 (52.5 s).

### `obbt="tree"` on m6

160 s → 62 s, against the 60 s limit.

### MINLPLib, 60 s cold, base and new run at the same time per instance (`--jobs 1 --threads-per-job 2` each, load 9–16)

| set | base solved | new solved | wrong | geometric mean time, both solved |
|---|---:|---:|---:|---|
| int50 | 35 | 37 | 0 / 0 | 26.2 → 23.9 s (35 instances) |
| start50 | 37 | 37 | 0 / 0 | 17.3 → 17.4 s (37 instances) |

- int50 gains: nvs01 and portfol_buyin. No losses.
- start50: node counts are identical on almost every instance.
- check_wrong seeds 0 and 1: no wrong answers.

## 4. What is left

- **Bound-kernel compile.** Starting it before `bound_domain`, with speculative options, would overlap its remaining 3–5 s. This is compile scheduling.
- **`_verify`** evaluates `f`, `ineq` and `eq` eagerly after the limit: up to 6 s on m426, the same eager-compile cause.
- **m6's `bound_domain`** still takes 10–14 s: feasible-point local solves, LP bounds, and a second local-solver build.
- **Node-box local solves** take 50–70% of the search time on the CLay models. This is the next lever for m399, m406 and m474.
- **Root dives on CLay** still take 4–5 s for useless incumbents.
