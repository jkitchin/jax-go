"""Figures: (1) relaxations tightening under subdivision, (2) B&B convergence."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxglobal import minimize, relax
from jaxglobal.problems import PROBLEMS, gramacy_lee

INK, MUTED, BLUE, ORANGE, AQUA, YELLOW = "#1a1a19", "#77766f", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6})


def relaxation_figure(path="figures/relaxations.png"):
    rf = jax.jit(jax.vmap(relax(gramacy_lee), in_axes=(0, None, None)))
    xs = np.linspace(0.5, 2.5, 801)
    F = np.asarray(jax.vmap(gramacy_lee)(jnp.asarray(xs[:, None])))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, nbox in zip(axes, [1, 4, 16]):
        edges = np.linspace(0.5, 2.5, nbox + 1)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            sel = (xs >= a) & (xs <= b)
            m = rf(jnp.asarray(xs[sel, None]), jnp.array([a]), jnp.array([b]))
            lab = k == 0
            ax.fill_between(xs[sel], np.asarray(m.cv), np.asarray(m.cc), color=BLUE, alpha=0.12, lw=0)
            ax.plot(xs[sel], m.cv, color=BLUE, lw=2, label="convex relaxation cv" if lab else None)
            ax.plot(xs[sel], m.cc, color=ORANGE, lw=2, label="concave relaxation cc" if lab else None)
            ax.hlines(float(m.lo[0]), a, b, color=MUTED, lw=1.2, ls="--", label="interval bound" if lab else None)
        ax.plot(xs, F, color=INK, lw=1.5, label="f(x)")
        ax.set_title(f"{nbox} box{'es' if nbox > 1 else ''}", color=INK, loc="left")
        ax.set_xlabel("x")
        ax.set_ylim(-3, 6)
    axes[0].set_ylabel("value")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Gramacy–Lee: sin(10πx)/(2x) + (x−1)⁴ — McCormick/αBB relaxations derived from the jaxpr",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def convergence_figure(path="figures/convergence.png"):
    names = ["shubert", "eggholder", "hartmann6", "goldstein_price"]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in zip(names, colors):
        p = next(q for q in PROBLEMS if q.name == name)
        r = minimize(p.f, p.lo, p.hi, abs_tol=1e-5, rel_tol=1e-6)
        h = np.array(r.history)
        gap = np.maximum(h[:, 3] - h[:, 2], 1e-7)
        ax.plot(h[:, 1], gap, color=c, lw=2, label=f"{name} (n={p.lo.size})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes evaluated")
    ax.set_ylabel("certified gap  UB − LB")
    ax.set_title("Branch-and-bound convergence", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    import os

    os.makedirs("figures", exist_ok=True)
    relaxation_figure()
    convergence_figure()
