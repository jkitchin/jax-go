# Solver options

Everything is controlled by keyword arguments of `minimize`. The defaults are
the configuration used for the published results; most problems need none of
them changed beyond tolerances and a time limit.

```python
r = minimize(f, lo, hi, ineq=g, eq=h, integer=None,
             abs_tol=1e-6, rel_tol=1e-6, time_limit=60, branch_var="width")
```

## Problem

| argument | default | meaning |
|---|---|---|
| `f` | | objective, a JAX-traceable function of a 1-D array returning a scalar |
| `lo`, `hi` | | variable bounds; may be infinite (see [Modelling](modelling.md#bounds)) |
| `ineq` | `None` | `g(x) <= 0`, a scalar or 1-D array |
| `eq` | `None` | `h(x) == 0`, a scalar or 1-D array |
| `integer` | `None` | integer variables, as indices or a boolean mask |

## Termination and tolerances

| argument | default | meaning |
|---|---|---|
| `abs_tol`, `rel_tol` | `1e-6`, `1e-6` | stop with `"optimal"` when `fun - lower_bound <= max(abs_tol, rel_tol * abs(fun))` |
| `feas_tol` | `1e-6` | a point is feasible when `ineq <= feas_tol` and `abs(eq) <= feas_tol` |
| `max_iter` | `5000` | iterations of the branch-and-bound loop (status `"max_iter"`) |
| `time_limit` | `inf` | seconds, counted from the call, including tracing and compilation (status `"time_limit"`). If the kernels are not ready in time the result has `lower_bound=-inf` and `message` says so. |
| `verbose` | `False` | print progress every 25 iterations |

## Branching

Each iteration selects the boxes with the lowest bounds and splits them. Two
arguments decide how.

### `branch_var`: which variable to split

| value | rule |
|---|---|
| `"width"` (default) | the variable with the largest relative width among the variables that appear nonlinearly. On the lifted quadratic LP, relative width times the McCormick gap of the variable's products at the relaxation point. |
| `"pseudocost"` | the product score of the per-variable bound gains observed so far |
| `"reliability"` | like `"pseudocost"`, but variables with fewer than 2 observations are strong-branched (up to 4 of them per box), within the `batch` budget |
| `"strong"` | every selected parent is split on its best-scored variable (pseudocost product score, width before any pseudocost is known), and the padding slots of the kernel calls the iteration makes anyway strong-branch further candidates. Only variables at least 1/10 as wide as the widest are candidates. Each parent is then split on its 2 best candidates at once (those scoring at least half the best) into up to 4 orthants, each bounded by the largest of the tightened halves. |

Only variables that appear nonlinearly, detected from sampled Hessians, are
branched on in a continuous problem. Integer variables are split at integer
values.

On the 21 built-in problems plus four larger ones (all certified), node counts
relative to `"width"` as geometric means were 0.97 for `"pseudocost"`, 0.90
for `"reliability"` and 0.80 for `"strong"` (iterations 0.82). The largest
gains for `"strong"` were on hartmann3 (157 → 46 nodes), hartmann6
(1099 → 634) and a reactor network (1053 → 654). On problems with 2 to 6
variables the choice of variable matters little. `"strong"` is opt-in so that
the default reproduces the published result tables. `"reliability"` can be
several times slower in wall time on integer problems, because each
strong-branching trial is a bounded box. `Result.evaluations` counts every box
bounded, including strong-branching trials, so compare `evaluations` as well
as `nodes` when trying these rules.

### `branch_point`: where to split it

| value | rule |
|---|---|
| `"mid"` (default) | bisect |
| `"relax"` | split at the relaxation minimizer, pulled toward the midpoint and kept away from the faces |

## Bound tightening

| argument | default | meaning |
|---|---|---|
| `fbbt` | `True` | forward-backward interval propagation over the jaxpr before every node is bounded |
| `fbbt_rounds` | `5` | FBBT sweeps per node |
| `dbbt` | `True` | duality-based range reduction from each node's bound certificate `f >= lb + Σ r_j (x_j - lo_j)` |
| `obbt` | `"root"` | LP-based tightening of every variable: `"root"`, `"tree"` (every node) or `False` |
| `obbt_rounds` | `2` | OBBT rounds |

## Relaxations and node bounds

| argument | default | meaning |
|---|---|---|
| `univariate` | `True` | replace the relaxation of every smooth sub-expression of a single variable by a grid-hull envelope (nearly exact) |
| `alphabb` | `True` | whole-function αBB with scaled-Gershgorin α from an interval Hessian for smooth functions; `"eigen"` adds the eigenvalue relaxation of exact quadratics |
| `multivariate` | `False` | relax products by the multivariate McCormick rule of [Tsoukalas & Mitsos 2014](../references.md#ref-tsoukalas2014multivariate), tighter where a factor's interval straddles zero |
| `lifted` | `False` | bound nodes with the extended-space LP (one column per nonlinear jaxpr node) instead of the reduced-space cuts; opt-in and slower |
| `quadratic` | `True` | a constrained problem whose functions are all quadratic is bounded, and OBBT-tightened, over a lifted sparse LP with one McCormick-linked column per product |
| `crown` | `False` | add the CROWN backward linear bound of `f` to every node bound: `False`, `True`, or `"auto"` (on when every primitive of `f` has a CROWN rule and `f` has a saturating activation or ReLU). Useful for neural-network surrogates. |
| `n_fw` | `20` | Frank–Wolfe steps on the convex relaxation, which give the affine cuts |
| `n_kelley` | `4` | Kelley points added to the cut LP of a constrained problem |
| `safe` | `False` | round every quantity that feeds a lower bound outward by a few ulps (about 1.7× slower); see [Correctness](correctness.md) |

## Upper bounds (local solves)

| argument | default | meaning |
|---|---|---|
| `local_solve` | `True` | run local NLP solves for incumbents |
| `local_solver` | `"auto"` | `"pounce"`, `"optax"` (pure JAX), `"scipy"` (L-BFGS-B / SLSQP), or `"auto"` (pounce if installed) |
| `n_starts` | `1` | scales the number of local starts drawn from the relaxation minimizers of the most promising boxes |

Once there is an incumbent, local searches and integer dives that fail to
improve it are run less often.

## Batching and devices

| argument | default | meaning |
|---|---|---|
| `batch` | `256` | most boxes bounded per iteration |
| `batch_sizes` | `(16,)` | sizes of the kernel calls; a batch is padded to the smallest size that holds it or cut into calls of the largest. Each size is traced and compiled once, so the default is a single small size. |
| `devices` | `None` | JAX devices to split the largest calls across; e.g. `devices=jax.devices()[:1]` disables splitting |

On CPU, expose several cores as devices by setting
`XLA_FLAGS=--xla_force_host_platform_device_count=N` before JAX is imported.

The bounding kernels compile in background threads and are cached on disk
across processes, under `~/.cache/jaxglobal` by default. `JAXGLOBAL_CACHE_DIR`
moves the cache and `JAXGLOBAL_CACHE=0` turns it off.

## The result

`minimize` returns a `Result`:

| field | meaning |
|---|---|
| `x` | best point found |
| `fun` | `f(x)`, an upper bound on the global minimum |
| `lower_bound` | proven lower bound on the global minimum |
| `gap` | `fun - lower_bound` |
| `status` | `"optimal"`, `"infeasible"`, `"max_iter"`, `"time_limit"`, `"unbounded_domain"` or `"unverified"` |
| `message` | details for a non-optimal status |
| `converged` | whether the gap closed |
| `nodes`, `iterations`, `evaluations` | search-tree nodes, B&B iterations, boxes bounded (including strong-branching trials) |
| `time` | wall-clock seconds |
| `history` | per-iteration progress |
| `max_violation`, `verified` | the incumbent re-checked outside `jit` in float64: box, `ineq`, `eq`, integrality and `f(x) == fun`. A failed check turns `"optimal"` into `"unverified"`. |

See the [API reference](../api/jaxglobal.md) for the full docstring.
