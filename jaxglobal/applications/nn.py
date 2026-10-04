"""Neural-network-surrogate benchmark after Schweidtmann & Mitsos [schweidtmann2019deterministic].

A. M. Schweidtmann and A. Mitsos, "Deterministic global optimization with
artificial neural networks embedded", J. Optim. Theory Appl. 180(3), 925-948
(2019), doi:10.1007/s10957-018-1396-0 (arXiv:1801.07114).  Section 5.1: the
MATLAB ``peaks`` function on D = [-3,3]^2 (global min -6.551 at (0.228,-1.626))
is learned by tanh MLPs trained on Latin-hypercube samples (500 points, one
hidden layer of 47 neurons for the illustrative example; 2000 points for the
scaling study: one hidden layer with 40..700 neurons, and 1..6 hidden layers of
20 neurons).  The paper trains in MATLAB (Levenberg-Marquardt, early stopping)
and does not publish weights, so ours are RETRAINED in JAX (benchmarks/train_nn.py)
and the networks are *not* the paper's.  The paper has no higher-input-dimension
peaks cases (its other studies are application ANNs with data we do not have).

Weights live in ``data/<name>.npz`` (W0,b0,...,W{L},b{L}); the net is
x -> tanh(x W0 + b0) -> ... -> linear output.  Pyomo builders: reduced space
(``pyomo_reduced``, tanh expressions, no intermediate variables) and OMLT full
space (``pyomo_omlt``).
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import AppProblem

DATA = Path(__file__).parent / "data"
DOMAIN = 3.0  # peaks domain [-3, 3]^2

# (name, hidden layers, neurons, n_samples, in_paper_grid)
GRID = [
    ("peaks_1x5", 1, 5, 2000, False),    # extra-small, for smoke tests
    ("peaks_1x10", 1, 10, 2000, False),
    ("peaks_1x20", 1, 20, 2000, False),
    ("peaks_1x40", 1, 40, 2000, True),   # Fig. 2a, smallest width
    ("peaks_1x47", 1, 47, 500, True),    # Sec. 5.1 illustrative network
    ("peaks_1x100", 1, 100, 2000, True),
    ("peaks_1x200", 1, 200, 2000, True),
    ("peaks_1x700", 1, 700, 2000, True),  # Fig. 2a, largest width
    ("peaks_2x20", 2, 20, 2000, True),   # Fig. 2b: 20 neurons, 1..6 layers
    ("peaks_3x20", 3, 20, 2000, True),
    ("peaks_4x20", 4, 20, 2000, True),
    ("peaks_5x20", 5, 20, 2000, True),
    ("peaks_6x20", 6, 20, 2000, True),
]


def peaks(x1, x2):
    """MATLAB peaks(); works for numpy and jax arrays."""
    xp = jnp if isinstance(x1, jnp.ndarray) else np
    return (3 * (1 - x1) ** 2 * xp.exp(-x1**2 - (x2 + 1) ** 2)
            - 10 * (x1 / 5 - x1**3 - x2**5) * xp.exp(-x1**2 - x2**2)
            - xp.exp(-(x1 + 1) ** 2 - x2**2) / 3)


def load_weights(name):
    z = np.load(DATA / f"{name}.npz")
    n = len(z.files) // 2
    return [z[f"W{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)]


def forward_np(Ws, bs, x):
    """Plain numpy forward pass (float64) for a single point or batch."""
    h = np.asarray(x, dtype=float)
    for W, b in zip(Ws[:-1], bs[:-1]):
        h = np.tanh(h @ W + b)
    return (h @ Ws[-1] + bs[-1])[..., 0]


def make_f(Ws, bs):
    Wj = [jnp.asarray(W) for W in Ws]
    bj = [jnp.asarray(b) for b in bs]

    def f(x):
        h = x
        for W, b in zip(Wj[:-1], bj[:-1]):
            h = jnp.tanh(h @ W + b)
        return (h @ Wj[-1] + bj[-1])[0]

    return f


def _tanh_fn(act):
    """tanh as a Pyomo expression.  act="tanh": pyo.tanh (accepted by MAiNGO-free tools such as Ipopt);
    act="exp": 1 - 2/(exp(2z)+1), for solvers whose NL/GAMS interface has no tanh (SCIP, Couenne, BARON)."""
    import pyomo.environ as pyo

    if act == "tanh":
        return pyo.tanh
    return lambda z: 1 - 2 / (pyo.exp(2 * z) + 1)


def pyomo_reduced(Ws, bs, lo, hi, act="tanh"):
    """Reduced-space model: objective is one nested tanh expression of the inputs (see ``_tanh_fn`` for ``act``)."""
    import pyomo.environ as pyo

    th = _tanh_fn(act)

    m = pyo.ConcreteModel()
    n = len(lo)
    m.x = pyo.Var(range(n), bounds=lambda m, i: (float(lo[i]), float(hi[i])), initialize=0.0)
    h = [m.x[i] for i in range(n)]
    for W, b in zip(Ws[:-1], bs[:-1]):
        h = [th(sum(float(W[i, j]) * h[i] for i in range(W.shape[0])) + float(b[j]))
             for j in range(W.shape[1])]
    W, b = Ws[-1], bs[-1]
    m.obj = pyo.Objective(expr=sum(float(W[i, 0]) * h[i] for i in range(W.shape[0])) + float(b[0]))
    return m


def pyomo_omlt(Ws, bs, lo, hi, act="tanh"):
    """OMLT full-space formulation (FullSpaceSmoothNNFormulation, tanh).  Requires `omlt`.

    Input variables are ``m.nn.inputs[i]``; the objective is named ``obj``.
    """
    import pyomo.environ as pyo
    from omlt import OmltBlock
    from omlt.neuralnet import FullSpaceSmoothNNFormulation, NetworkDefinition
    from omlt.neuralnet.layer import DenseLayer, InputLayer

    n = len(lo)
    net = NetworkDefinition(scaled_input_bounds={i: (float(lo[i]), float(hi[i])) for i in range(n)})
    prev = InputLayer([n])
    net.add_layer(prev)
    for k, (W, b) in enumerate(zip(Ws, bs)):
        last = k == len(Ws) - 1
        layer = DenseLayer([W.shape[0]], [W.shape[1]], activation="linear" if last else "tanh",
                           weights=np.asarray(W), biases=np.asarray(b))
        net.add_layer(layer)
        net.add_edge(prev, layer)
        prev = layer
    m = pyo.ConcreteModel()
    m.nn = OmltBlock()
    form = FullSpaceSmoothNNFormulation(net)
    if act != "tanh":
        from omlt.neuralnet.activations.smooth import smooth_monotonic_activation_constraint

        th = _tanh_fn(act)
        # OMLT has no public hook on the smooth formulation; override its tanh constraint generator
        form._activation_constraints["tanh"] = lambda nb, n, lb, ly: smooth_monotonic_activation_constraint(nb, n, lb, ly, th)
    m.nn.build_formulation(form)
    m.obj = pyo.Objective(expr=m.nn.outputs[0])
    return m


def omlt_model(name, act="tanh"):
    """OMLT full-space Pyomo model for the named network."""
    Ws, bs = load_weights(name)
    return pyomo_omlt(Ws, bs, np.full(2, -DOMAIN), np.full(2, DOMAIN), act=act)


def reduced_model(name, act="tanh"):
    Ws, bs = load_weights(name)
    return pyomo_reduced(Ws, bs, np.full(2, -DOMAIN), np.full(2, DOMAIN), act=act)


# best certified global minimum found across solvers (filled from smoke runs); None = not certified
FSTAR = {"peaks_1x5": -2.88752, "peaks_1x10": -4.2516, "peaks_1x20": -6.32581}  # jaxglobal + BARON agree, gap closed (6 digits)


def _problem(name, L, H, nsamp, in_paper):
    Ws, bs = load_weights(name)
    lo, hi = np.full(2, -DOMAIN), np.full(2, DOMAIN)
    src = ("Retrained (JAX/Adam, seeded; benchmarks/train_nn.py) stand-in for Schweidtmann & Mitsos "
           f"[schweidtmann2019deterministic] (Sec. 5.1): tanh MLP {L}x{H} on peaks over [-3,3]^2, {nsamp} LHC samples. "
           "NOT the paper's weights.")
    note = f"{L} hidden layer(s) x {H} tanh neurons; " + ("architecture in paper grid" if in_paper else "extra small size, not in paper")
    return AppProblem(name, make_f(Ws, bs), lo, hi, None, None, FSTAR.get(name), src,
                      lambda Ws=Ws, bs=bs, lo=lo, hi=hi: pyomo_reduced(Ws, bs, lo, hi), note)


PROBLEMS = [_problem(name, L, H, ns, ip) for name, L, H, ns, ip in GRID if (DATA / f"{name}.npz").exists()]
