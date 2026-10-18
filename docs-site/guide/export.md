# Export to SCIP and `.nl`

`jaxglobal.export` takes a problem in the same form `minimize` takes it (`f`,
`lo`, `hi`, `ineq`, `eq`, `integer`) and builds a PySCIPOpt model or writes an
AMPL `.nl` file for Couenne, SCIP, BARON or `jaxglobal.nl.read_nl`. You do not
write a Pyomo twin or reformulate anything by hand. The functions are traced,
`scan`/`fori_loop` and fixed-count `while_loop`s are unrolled, and `jit` and
`custom_jvp` calls are inlined. Every operation is then translated exactly,
replaced by an exact identity, reformulated with auxiliary variables, or
refused with an `ExportError` that names it. The report's `approximated` field
lists the few things that are not exact (see below). Evaluate `f` with JAX at
the solver's `x` before you trust its objective value.
`examples/export_scip.py` in the repository exports a model that uses
`where`, `maximum`, `tanh`, `relu`, a `fori_loop` and an integer variable, solves
it with SCIP, and compares the result with `minimize`.

```python
from jaxglobal import export

m = export.to_scip(f, lo, hi, ineq=g, eq=h, integer=[2])   # a pyscipopt.Model
print(m.data.report)                  # what happened to each operation
m.setParam("limits/time", 60); m.optimize()
x = export.solution(m)                # best point, in the order of x

r = export.to_nl(f, lo, hi, "model.nl", ineq=g)   # returns the report
# r.var_order[k] is the variable of .nl column k (integers are written last)
export.to_nl(f, lo, hi, "portable.nl", ineq=g, portable=True)  # readable by the scip and couenne binaries
export.inspect(f, lo, hi, target="scip")          # report only; does not raise
```

| | SCIP (`to_scip`) | `.nl` (`to_nl`) |
|---|---|---|
| exact | `+ - * /`, neg, sum, dot/matmul, reductions, `x**p` (constant `p`), exp, log, sqrt, abs, sin, cos; indexing, slicing, reshape, broadcast, concatenate, stack | the SCIP set, plus tan, tanh, sinh, cosh, atan, asin, acos, atanh, asinh, acosh, atan2, `x**y`; by default also max, min, `where` (`if`) and comparisons |
| exact rewrite | tanh → `1 - 2/(exp(2x)+1)` (or an overflow-free form, below), sigmoid, softplus, log1p, expm1, tan, sinh, cosh, asinh (odd form), acosh, atanh; `x**y` → `exp(y log x)` when `x > 0` on the box; max/min (and so relu, clip) → `(a + b ± \|a - b\|)/2` | sigmoid, log1p, expm1; sign as nested `if` |
| reformulated | `where`/`select_n` (aux `y` and four big-M rows); `<`, `<=`, `>`, `>=` (a binary); `sign` (two binaries); and/or/not; max/min with `minmax="bigm"`; shared subexpressions of 50+ operations (aux `w == expr`); product operands that pyscipopt would expand past 1000 monomials | the same as SCIP, with `nonsmooth="reformulate"` or `portable=True` |
| not exact (`approximated`) | reformulated comparisons and `sign` at ties (below); `reduce_precision` and float64→float32 casts (identities) | the same, when reformulated |
| refused | erf and other special functions, a `while_loop` whose condition depends on `x`, `lax.cond` on a model value, float→int conversion and indices computed from `x`, `==`/`!=` between model values, atan/asin/acos/atan2, a `where` branch undefined on part of the box (below), big-M constants ≥ 1e20, an intermediate value that reaches 1e20 over the box, division by the constant 0, any primitive not listed | the same, minus the inverse trig functions (refused again with `portable=True`) |

Big-M constants come from interval bounds of the operands over the box. A
reformulation is refused when a bound is not finite or a constant reaches 1e20
(SCIP's infinity). Above 1e8 the export warns (`ExportBigMWarning`) and adds a
note, because the solver's tolerances, multiplied by M, then decide which branch
is taken.

Points to check:

- **Ties.** A reformulated comparison cannot be strict. With `strict_eps=0`
  (the default), at a tie `u == v` either branch is allowed. Where the two
  branches of a `where` agree at the tie (relu, a continuous switch) that is
  harmless. For a piecewise-constant model, such as a step or `sign`, the model
  is a relaxation: the solver can return a breakpoint value that JAX never
  produces. `strict_eps > 0` excludes `0 < |u - v| < strict_eps` on the wrong
  side, which is a restriction instead. Pick it well above the solver's
  feasibility tolerance (SCIP: 1e-6). Either way, evaluate `f` with JAX at the
  returned `x`.
- **Branch domains.** The reformulation of `where` evaluates both branches
  everywhere, so `jnp.where(x > 0, jnp.sqrt(x), x)` would lose `x < 0`. The
  export refuses it and names the operation. Make the branch defined
  everywhere: `jnp.sqrt(jnp.maximum(x, 0))`. The double-`where` idiom
  `jnp.where(x > 0, jnp.sqrt(jnp.where(x > 0, x, 1.0)), x)` is also refused:
  the check uses interval bounds, which cannot see that the inner `where`
  keeps `sqrt`'s argument positive. Use the `maximum` form.
- **Overflow.** SCIP treats a value of 1e20 or more as infinity, and a point
  where any intermediate expression reaches it as an invalid evaluation, which
  it cuts off. The `tanh` and `sigmoid` rewrites therefore pick a form from the
  argument's interval in which no `exp` overflows (`exp(-2x)` when `x >= 0`;
  `exp(x)/(1 + exp(x))` when `x <= 0`; `exp(x - |x|)` and `exp(-x - |x|)`, both
  at most 1, when the interval straddles 0 with large `|x|`), and `asinh` /
  `acosh` avoid `x^2` for `|x| > 1e9`. All of these are exact. Any other
  intermediate whose interval bound reaches 1e20 while its operands are bounded
  (such as `exp(x)` for `x` up to 60) is refused; tighten the bounds.
- **Which `.nl` reader.** The default `.nl` keeps max, min, `if` and the
  inverse and hyperbolic functions as operators. `read_nl` reads them all, but
  the `scip` binary's reader takes only `+ - * /`, neg, pow, exp, log, sqrt,
  abs, sin, cos and sums. Couenne also takes tan, tanh, sinh and cosh. Neither
  takes min/max, `if`/comparisons, atan/asin/acos/atan2 or the inverse
  hyperbolic functions. `portable=True` writes only what both read. It
  rewrites and reformulates the rest as for SCIP, and it uses auxiliary
  variables instead of defined variables (`V` segments). Use `portable=True`
  with Couenne: Couenne 0.5.8 misreads `V` segments in general, even in
  files AMPL itself writes (six-hump camel gives -0.3232 instead of -1.0316).

**Reading the report**
(`model.data.report`, the return value of `to_nl`, or `inspect`):

- the first line counts the original and auxiliary variables (binaries among
  them), the original and added constraints, and the scalar operations;
- `exact`, `rewritten` and `reformulated` give per-operation counts, and the last
  two name the identity or construction used;
- `approximated` lists what is not exact and how (ties, precision casts). If it
  is not empty, evaluate `f` at the solver's `x`;
- `objective: epigraph` means the nonlinear objective became `f(x) <= t`;
- `refused` lists what blocked the export (`inspect` returns this instead of
  raising);
- `assumption` lists the simplifications that hold at finite points: `isnan` of
  a model value is False and `isfinite` is True;
- `note` lists unrolled iterations, dropped constant constraints, and the
  largest big-M constant when it is above 1e8.

`ExportSizeWarning` fires above `max_size` scalar operations (default 1e5).


See also the [API reference for `jaxglobal.export`](../api/export.md).
