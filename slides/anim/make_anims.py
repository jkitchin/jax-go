"""Generate the animated SVG figures used in slides/jaxglobal.md.

Each figure is one self-contained SVG with SMIL animations on a common loop.
Every loop starts on the finished picture (so a static PDF export, which
renders the first frame, still shows the whole explanation), then resets and
plays the steps. The numbers are computed here, not drawn by hand.

Run:  python slides/anim/make_anims.py
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
W, H = 960, 440
BLUE, LIGHT, ORANGE, RED, GREEN, GREY, INK = "#1c5cab", "#eef4fc", "#d9822b", "#c0392b", "#2e8b57", "#9aa3ad", "#1d1d1b"
FONT = "font-family='Helvetica, Arial, sans-serif'"
HOLD = 1.6  # seconds showing the finished picture at the start of each loop


class Timeline:
    """Builds <animate> elements on a shared loop of ``T`` seconds."""

    def __init__(self, T):
        self.T = T

    def anim(self, attr, final, frames, calc="linear"):
        """``frames``: [(t, value)] in play time (after the opening hold).

        The loop holds ``final`` for HOLD seconds, jumps to the first frame,
        plays, and ends on ``final`` again.
        """
        pts = [(0.0, final), (HOLD - 0.25, final)]
        pts += [(HOLD + t, v) for t, v in frames]
        if pts[-1][0] < self.T:
            pts.append((self.T, final))
        keys = ";".join(f"{min(t / self.T, 1.0):.5f}" for t, _ in pts)
        vals = ";".join(str(v) for _, v in pts)
        return (f"<animate attributeName='{attr}' dur='{self.T}s' repeatCount='indefinite' "
                f"keyTimes='{keys}' values='{vals}' calcMode='{calc}'/>")  # fmt: skip

    def show(self, t_on, t_off=None, fade=0.35):
        """Opacity: hidden, fade in at t_on (play time), optionally out at t_off."""
        fr = [(0.0, 0), (t_on, 0), (t_on + fade, 1)]
        if t_off is not None:
            fr += [(t_off, 1), (t_off + fade, 0)]
        else:
            fr += [(self.T - HOLD - 0.01, 1)]
        return self.anim("opacity", 1 if t_off is None else 0, fr)

    def caption(self, t_on, t_off):
        """Opacity for step captions: in the hold, only the summary caption shows."""
        # fade out within 0.15 s and fade in after 0.15 s, so consecutive captions never overlap
        return self.anim("opacity", 0, [(0.0, 0), (t_on + 0.15, 0), (t_on + 0.4, 1), (t_off, 1), (t_off + 0.15, 0)])


def svg(body, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' {FONT}>"
            f"<title>{title}</title><rect width='{W}' height='{H}' fill='white'/>{body}</svg>\n")  # fmt: skip


def text(x, y, s, size=18, color=INK, anchor="start", weight="normal", extra=""):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' text-anchor='{anchor}' "
            f"font-weight='{weight}'>{s}{extra}</text>")  # fmt: skip


def captions(tl, steps, summary, y=420):
    """Step captions at the bottom; ``summary`` is shown during the hold."""
    out = [f"<text x='{W / 2}' y='{y}' font-size='19' fill='{BLUE}' text-anchor='middle' font-weight='bold' opacity='1'>"
           f"{summary}{tl.anim('opacity', 1, [(0.0, 0), (tl.T - HOLD - 0.4, 0), (tl.T - HOLD, 1)])}</text>"]
    for t_on, t_off, s in steps:
        out.append(f"<text x='{W / 2}' y='{y}' font-size='19' fill='{INK}' text-anchor='middle' opacity='0'>"
                   f"{s}{tl.caption(t_on, t_off)}</text>")  # fmt: skip
    return "".join(out)


def poly_path(pts):
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts)


# ---------------------------------------------------------------------------
# FBBT: forward intervals up the expression graph, backward inversion down
# ---------------------------------------------------------------------------


def fbbt():
    T = 16.0
    tl = Timeline(T)
    # problem: x in [0, 5], y in [-2, 2], constraint x + y^2 <= 2
    r2 = math.sqrt(2)
    b = []
    b.append(text(30, 34, "constraint  x + y² ≤ 2,   box  x ∈ [0, 5],  y ∈ [−2, 2]", 19, INK, weight="bold"))

    # --- expression graph (left) -----------------------------------------
    nodes = {"x": (110, 300), "y": (330, 300), "sq": (330, 200), "plus": (220, 110)}
    edges = [("x", "plus"), ("sq", "plus"), ("y", "sq")]
    for a, c in edges:
        (x1, y1), (x2, y2) = nodes[a], nodes[c]
        b.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{GREY}' stroke-width='2'/>")
    labels = {"x": "x", "y": "y", "sq": "( )²", "plus": "+"}
    for k, (x, y) in nodes.items():
        b.append(f"<circle cx='{x}' cy='{y}' r='26' fill='{LIGHT}' stroke='{BLUE}' stroke-width='2'/>")
        b.append(text(x, y + 6, labels[k], 20, BLUE, "middle", "bold"))
    b.append(text(220, 58, "≤ 2", 20, RED, "middle", "bold"))
    b.append(f"<line x1='220' y1='64' x2='220' y2='84' stroke='{RED}' stroke-width='2'/>")

    # interval labels: (node, text, color, t_on, t_off)
    def lab(x, y, s, color, t_on, t_off=None, anchor="start"):
        return (f"<text x='{x}' y='{y}' font-size='18' fill='{color}' text-anchor='{anchor}' font-weight='bold' "
                f"opacity='{1 if t_off is None else 0}'>{s}{tl.show(t_on, t_off)}</text>")  # fmt: skip

    # forward
    b.append(lab(60, 350, "[0, 5]", BLUE, 0.3, 7.0))
    b.append(lab(290, 350, "[−2, 2]", BLUE, 0.3, 8.6))
    b.append(lab(368, 205, "[0, 4]", BLUE, 1.6, 7.6))
    b.append(lab(262, 115, "[0, 9]", BLUE, 3.0, 4.6))
    # constraint clamps the output
    b.append(lab(262, 115, "[0, 2]", RED, 4.6))
    # backward
    b.append(lab(60, 350, "[0, 2]", ORANGE, 7.0))
    b.append(lab(368, 205, "[0, 2]", ORANGE, 7.6))
    b.append(lab(290, 350, "[−1.41, 1.41]", ORANGE, 8.6))
    # arrows for the two passes
    up = (f"<g opacity='0'>{tl.show(0.3, 4.4)}<path d='M455,330 L455,130' stroke='{BLUE}' stroke-width='4' "
          f"marker-end='url(#ab)'/>{text(468, 230, 'forward', 17, BLUE)}</g>")
    down = (f"<g opacity='0'>{tl.show(6.0, 9.6)}<path d='M455,120 L455,320' stroke='{ORANGE}' stroke-width='4' "
            f"marker-end='url(#ao)'/>{text(468, 230, 'backward', 17, ORANGE)}</g>")
    b.append(up + down)
    b.append(text(30, 385, "x ∈ [0,2] − [0,4]  →  x ≤ 2;     y² ∈ [0,2] − [0,2]  →  |y| ≤ √2", 16, ORANGE,
                  extra=tl.show(7.0)))  # fmt: skip

    # --- the box in the (x, y) plane (right) ------------------------------
    px0, py0, s = 560, 210, 66  # origin pixel for x=0,y=0 and px per unit

    def P(x, y):
        return px0 + s * x, py0 - s * y

    # feasible region x <= 2 - y^2
    ys = np.linspace(-r2, r2, 60)
    region = [P(0, -r2)] + [P(2 - y * y, y) for y in ys] + [P(0, r2)]
    b.append(f"<path d='{poly_path(region)} Z' fill='{GREEN}' fill-opacity='0.25' stroke='{GREEN}' stroke-width='1.5'/>")
    b.append(text(*P(0.2, 0.45), "feasible", 15, GREEN))
    # axes
    b.append(f"<line x1='{P(0, -2.4)[0]}' y1='{P(0, -2.4)[1]}' x2='{P(0, 2.4)[0]}' y2='{P(0, 2.4)[1]}' stroke='{GREY}'/>")
    b.append(f"<line x1='{P(-0.3, 0)[0]}' y1='{P(-0.3, 0)[1]}' x2='{P(5.4, 0)[0]}' y2='{P(5.4, 0)[1]}' stroke='{GREY}'/>")
    b.append(text(*P(5.25, -0.35), "x", 16, GREY))
    b.append(text(*P(-0.35, 2.2), "y", 16, GREY))
    # box rectangle animates from [0,5]x[-2,2] to [0,2]x[-r2,r2]
    (ax, ay), (bx, by) = P(0, 2), P(5, -2)
    (cx_, cy_), (dx_, dy_) = P(0, r2), P(2, -r2)
    rect = (f"<rect x='{ax}' y='{cy_:.1f}' width='{dx_ - cx_:.1f}' height='{dy_ - cy_:.1f}' fill='none' "
            f"stroke='{BLUE}' stroke-width='3'>"
            + tl.anim("y", f"{cy_:.1f}", [(0, f"{ay:.1f}"), (9.8, f"{ay:.1f}"), (11.6, f"{cy_:.1f}")])
            + tl.anim("width", f"{dx_ - cx_:.1f}", [(0, f"{bx - ax:.1f}"), (9.8, f"{bx - ax:.1f}"), (11.6, f"{dx_ - cx_:.1f}")])
            + tl.anim("height", f"{dy_ - cy_:.1f}", [(0, f"{by - ay:.1f}"), (9.8, f"{by - ay:.1f}"), (11.6, f"{dy_ - cy_:.1f}")])
            + "</rect>")  # fmt: skip
    b.append(rect)
    removed = 1 - (2 * 2 * r2) / (5 * 4)
    b.append(text(*P(2.1, 1.65), f"{removed:.0%} of the box removed", 15, BLUE, extra=tl.show(11.6)))

    defs = (f"<defs><marker id='ab' markerWidth='16' markerHeight='16' refX='8' refY='8' orient='auto' markerUnits='userSpaceOnUse'>"
            f"<path d='M0,0 L16,8 L0,16 Z' fill='{BLUE}'/></marker>"
            f"<marker id='ao' markerWidth='16' markerHeight='16' refX='8' refY='8' orient='auto' markerUnits='userSpaceOnUse'>"
            f"<path d='M0,0 L16,8 L0,16 Z' fill='{ORANGE}'/></marker></defs>")  # fmt: skip
    caps = captions(tl, [
        (0.2, 4.4, "1 · forward: interval of every node, leaves to root"),
        (4.5, 6.0, "2 · the constraint clamps the root: [0, 9] → [0, 2]"),
        (6.1, 9.7, "3 · backward: invert each operation to narrow its inputs"),
        (9.8, T - HOLD - 0.5, "4 · the box shrinks before any branching"),
    ], "FBBT: forward–backward interval propagation on the jaxpr, every node")  # fmt: skip
    return svg(defs + "".join(b) + caps, "FBBT animation")


# ---------------------------------------------------------------------------
# DBBT: the bound certificate is an affine minorant; it cannot beat UB past x*
# ---------------------------------------------------------------------------


def dbbt():
    T = 14.0
    tl = Timeline(T)

    def f(x):
        return 1.2 + 0.55 * np.sin(1.3 * x) + 0.02 * (x - 1.5) ** 2

    lo, hi = 0.0, 10.0
    xs = np.linspace(lo, hi, 400)
    fx = f(xs)
    r = 0.22
    lb = float(np.min(fx - r * (xs - lo)))  # f >= lb + r (x - lo) on the box
    ub = float(np.min(fx)) + 0.25
    xstar = lo + (ub - lb) / r
    ymin, ymax = min(lb, 0.0) - 0.1, max(float(fx.max()), lb + r * (hi - lo)) + 0.1
    X0, sx = 70, 80
    sy = 270 / (ymax - ymin)
    Y0 = 335 + sy * ymin  # pixel of y = 0

    def P(x, y):
        return X0 + sx * x, Y0 - sy * y

    b = [text(30, 34, "after bounding a node:  f(x) ≥ lb + r·(x − lo)   (from the cut / LP duals)", 19, INK, weight="bold")]
    b.append(f"<line x1='{P(lo, 0)[0]}' y1='{P(lo, 0)[1]}' x2='{P(hi, 0)[0]}' y2='{P(hi, 0)[1]}' stroke='{GREY}'/>")
    b.append(f"<path d='{poly_path([P(x, y) for x, y in zip(xs, fx)])}' fill='none' stroke='{INK}' stroke-width='2.5'/>")
    b.append(text(*P(9.4, f(9.4) + 0.2), "f", 18, INK, weight="bold"))
    # incumbent
    b.append(f"<g opacity='1'>{tl.show(1.0)}<line x1='{P(lo, ub)[0]}' y1='{P(lo, ub)[1]}' x2='{P(hi, ub)[0]}' "
             f"y2='{P(hi, ub)[1]}' stroke='{GREEN}' stroke-width='2' stroke-dasharray='8 5'/>"
             f"{text(P(hi, ub)[0] - 4, P(hi, ub)[1] - 8, f'incumbent UB = {ub:.2f}', 16, GREEN, 'end')}</g>")
    # the cut line, drawn from left to right
    cut = [P(lo, lb), P(hi, lb + r * (hi - lo))]
    length = math.dist(*cut)
    b.append(f"<line x1='{cut[0][0]:.1f}' y1='{cut[0][1]:.1f}' x2='{cut[1][0]:.1f}' y2='{cut[1][1]:.1f}' "
             f"stroke='{BLUE}' stroke-width='3' stroke-dasharray='{length:.1f}' stroke-dashoffset='0'>"
             + tl.anim("stroke-dashoffset", 0, [(0, f"{length:.1f}"), (2.6, f"{length:.1f}"), (4.6, 0)])
             + "</line>")  # fmt: skip
    b.append(text(*P(1.0, lb + r * 1.0 - 0.28), "lb + r·(x − lo)", 16, BLUE, extra=tl.show(4.2)))
    # intersection and removed region
    sxp, syp = P(xstar, ub)
    b.append(f"<circle cx='{sxp:.1f}' cy='{syp:.1f}' r='7' fill='{RED}' opacity='1'>{tl.show(5.4)}</circle>")
    b.append(text(sxp + 14, syp + 26, "x* = lo + (UB − lb)/r", 16, RED, "start", "bold", extra=tl.show(5.6)))
    (rx, ry), (rx2, ry2) = P(xstar, ymax), P(hi, ymin + 0.05)
    b.append(f"<rect x='{rx:.1f}' y='{ry:.1f}' width='{rx2 - rx:.1f}' height='{ry2 - ry:.1f}' fill='{GREY}' "
             f"fill-opacity='0.35' opacity='1'>{tl.show(7.0)}</rect>")
    b.append(text((rx + rx2) / 2, ry + 26, "f > UB here: cannot improve", 16, INK, "middle", extra=tl.show(7.2)))
    # the node's range bracket shrinks
    by = 352
    x_lo_px, x_hi_px, x_new = P(lo, 0)[0], P(hi, 0)[0], sxp
    b.append(f"<line x1='{x_lo_px}' y1='{by}' x2='{x_new:.1f}' y2='{by}' stroke='{BLUE}' stroke-width='6'>"
             + tl.anim("x2", f"{x_new:.1f}", [(0, x_hi_px), (8.4, x_hi_px), (10.0, f"{x_new:.1f}")]) + "</line>")
    b.append(text(x_lo_px, by + 26, "lo", 15, BLUE, "middle"))
    b.append(text(x_new, by + 26, "new hi", 15, BLUE, "middle", extra=tl.show(10.0)))
    caps = captions(tl, [
        (0.2, 2.4, "1 · an incumbent gives the cutoff f ≤ UB"),
        (2.5, 5.2, "2 · the node's bound certificate is an affine minorant of f"),
        (5.3, 8.3, "3 · past x* the minorant already exceeds UB"),
        (8.4, T - HOLD - 0.5, "4 · so x > x* is cut from the node: free, from duals already computed"),
    ], "DBBT: duality-based range reduction reuses every node's certificate")  # fmt: skip
    return svg("".join(b) + caps, "DBBT animation")


# ---------------------------------------------------------------------------
# OBBT: min / max each variable over the LP outer approximation
# ---------------------------------------------------------------------------


def _clip(poly, a, b, c):
    """Sutherland-Hodgman: keep a*x + b*y <= c."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def obbt():
    T = 17.0
    tl = Timeline(T)
    # true set: disk radius 2, x + y >= 1, y <= 1.3 + 0.35 sin(3x)  (nonconvex top)
    poly = [(-3, -3), (3, -3), (3, 3), (-3, 3)]
    for k in range(12):  # tangent cuts of the disk
        th = 2 * math.pi * k / 12 + 0.13
        poly = _clip(poly, math.cos(th), math.sin(th), 2.0)
    poly = _clip(poly, -1, -1, -1.0)  # x + y >= 1
    poly = _clip(poly, 0, 1, 1.65)  # y <= max of the sin cap (interval bound)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)

    X0, Y0, s = 430, 222, 52

    def P(x, y):
        return X0 + s * x, Y0 - s * y

    b = [text(30, 34, "relaxation LP: convex cuts of every constraint;  2n LPs:  min / max  xᵢ", 19, INK, weight="bold")]
    # true nonconvex set via nested clip paths
    disk = f"<clipPath id='disk'><circle cx='{X0}' cy='{Y0}' r='{2 * s}'/></clipPath>"
    half = f"<clipPath id='half'><path d='{poly_path([P(-3, 4), P(4, -3), P(4, 4)])} Z'/></clipPath>"
    xx = np.linspace(-3, 3, 120)
    cap = [P(-3, -3)] + [P(x, 1.3 + 0.35 * math.sin(3 * x)) for x in xx] + [P(3, -3)]
    capc = f"<clipPath id='cap'><path d='{poly_path(cap)} Z'/></clipPath>"
    b.append(f"<defs>{disk}{half}{capc}</defs>")
    b.append(f"<g clip-path='url(#cap)'><g clip-path='url(#half)'><rect x='{P(-3, 3)[0]}' y='{P(-3, 3)[1]}' "
             f"width='{6 * s}' height='{6 * s}' fill='{GREEN}' fill-opacity='0.3' clip-path='url(#disk)'/></g></g>")
    b.append(text(*P(2.25, -0.3), "feasible set", 15, GREEN))
    b.append(f"<line x1='{P(2.2, -0.2)[0]}' y1='{P(2.2, -0.2)[1]}' x2='{P(1.55, 0.2)[0]}' y2='{P(1.55, 0.2)[1]}' stroke='{GREEN}'/>")
    # LP polygon
    b.append(f"<path d='{poly_path([P(*p) for p in poly])} Z' fill='{BLUE}' fill-opacity='0.08' stroke='{BLUE}' "
             f"stroke-width='2' stroke-dasharray='6 4' opacity='1'>{tl.show(1.0)}</path>")
    b.append(text(*P(-2.9, -2.75), "blue dashed: LP outer approximation", 15, BLUE, extra=tl.show(1.4)))
    b.append(f"<rect x='{P(-3, 3)[0]}' y='{P(-3, 3)[1]}' width='{6 * s}' height='{6 * s}' fill='none' "
             f"stroke='{GREY}' stroke-width='1.5' stroke-dasharray='4 4'/>")
    # original box [-3, 3]^2 shrinking to the LP bounding box
    (ax, ay), (cx_, cy_) = P(-3, 3), P(bx0, by1)
    W0, W1, H1 = 6 * s, (bx1 - bx0) * s, (by1 - by0) * s
    b.append(f"<rect x='{cx_:.1f}' y='{cy_:.1f}' width='{W1:.1f}' height='{H1:.1f}' fill='none' stroke='{INK}' stroke-width='3'>"
             + tl.anim("x", f"{cx_:.1f}", [(0, ax), (12.0, ax), (13.6, f"{cx_:.1f}")])
             + tl.anim("y", f"{cy_:.1f}", [(0, ay), (12.0, ay), (13.6, f"{cy_:.1f}")])
             + tl.anim("width", f"{W1:.1f}", [(0, W0), (12.0, W0), (13.6, f"{W1:.1f}")])
             + tl.anim("height", f"{H1:.1f}", [(0, W0), (12.0, W0), (13.6, f"{H1:.1f}")])
             + "</rect>")  # fmt: skip

    # four sweeping objective lines
    def sweep(vertical, start, stop, t0, label):
        """An objective level line moving from the box edge to the LP optimum; fades when the box shrinks."""
        if vertical:
            a, e = P(start, 0)[0], P(stop, 0)[0]
            fixed = f"y1='{P(0, 3)[1]}' y2='{P(0, -3)[1]}'"
            attrs, lab = ("x1", "x2"), ("x", P(0, 3)[1] - 8)
        else:
            a, e = P(0, start)[1], P(0, stop)[1]
            fixed = f"x1='{P(-3, 0)[0]}' x2='{P(3, 0)[0]}'"
            attrs, lab = ("y1", "y2"), ("y", None)
        move = [(0, f"{a:.1f}"), (t0, f"{a:.1f}"), (t0 + 1.6, f"{e:.1f}")]
        ln = f"<line {fixed} {attrs[0]}='{e:.1f}' {attrs[1]}='{e:.1f}' stroke='{ORANGE}' stroke-width='3'>"
        ln += tl.anim(attrs[0], f"{e:.1f}", move) + tl.anim(attrs[1], f"{e:.1f}", move) + "</line>"
        if vertical:
            t = f"<text y='{lab[1]:.1f}' x='{e:.1f}' font-size='15' fill='{ORANGE}' text-anchor='middle'>{label}"
            t += tl.anim("x", f"{e:.1f}", move) + "</text>"
        else:
            t = f"<text x='{P(3, 0)[0] + 8:.1f}' y='{e + 5:.1f}' font-size='15' fill='{ORANGE}'>{label}"
            t += tl.anim("y", f"{e + 5:.1f}", [(k, f"{float(v) + 5:.1f}") for k, v in move]) + "</text>"
        return f"<g opacity='0'>{tl.show(t0, 11.8)}{ln}{t}</g>"

    b.append(sweep(True, -3, bx0, 2.6, "min x"))
    b.append(sweep(True, 3, bx1, 4.8, "max x"))
    b.append(sweep(False, -3, by0, 7.0, "min y"))
    b.append(sweep(False, 3, by1, 9.2, "max y"))
    b.append(text(*P(4.3, 1.4), f"x ∈ [{bx0:.2f}, {bx1:.2f}]", 17, INK, weight="bold", extra=tl.show(13.6)))
    b.append(text(*P(4.3, 0.9), f"y ∈ [{by0:.2f}, {by1:.2f}]", 17, INK, weight="bold", extra=tl.show(13.6)))
    b.append(text(*P(4.3, 0.4), "was [−3, 3] × [−3, 3]", 15, GREY, extra=tl.show(13.6)))
    caps = captions(tl, [
        (0.2, 2.5, "1 · cuts of the convex relaxations give an LP that contains the feasible set"),
        (2.6, 11.9, "2 · minimize and maximize each variable over it: 2n LPs, one vmap"),
        (12.0, T - HOLD - 0.5, "3 · each result is a weak-duality certificate, so the new box is valid"),
    ], "OBBT: optimality-based bound tightening (root, or every node)")  # fmt: skip
    return svg("".join(b) + caps, "OBBT animation")


# ---------------------------------------------------------------------------
# alphaBB: add a convex quadratic until the curvature is nonnegative
# ---------------------------------------------------------------------------


def alphabb():
    T = 15.0
    tl = Timeline(T)

    def f(x):
        return np.sin(1.5 * x) + 0.2 * x

    lo, hi = 0.0, 4.0
    alpha = 2.25  # max(0, -min f'') on [0, 4];  f'' = -2.25 sin(1.5 x)
    xs = np.linspace(lo, hi, 161)
    X0, Y0, sx, sy = 90, 118, 150, 52

    def P(x, y):
        return X0 + sx * x, Y0 - sy * y

    RX = P(hi, 0)[0] + 40  # right-hand readouts

    def curve(a, a_lo=lo, a_hi=hi, n=161):
        z = np.linspace(a_lo, a_hi, n)
        return poly_path([P(x, y) for x, y in zip(z, f(z) + 0.5 * a * (z - a_lo) * (z - a_hi))])

    b = [text(30, 34, "f(x) + (α/2)(x − l)(x − u)  is convex when  α ≥ max(0, −min f″)   (f″ from interval jax.grad(jax.grad))",
              17, INK, weight="bold")]  # fmt: skip
    b.append(f"<line x1='{P(lo, 0)[0]}' y1='{P(lo, 0)[1]}' x2='{P(hi, 0)[0]}' y2='{P(hi, 0)[1]}' stroke='{GREY}'/>")
    b.append(f"<path d='{curve(0)}' fill='none' stroke='{INK}' stroke-width='2.5'/>")
    b.append(text(*P(4.03, f(4.0)), "f", 18, INK, weight="bold"))
    # whole-box underestimator: morph alpha 0 -> alpha*
    stages = [0, 0.5, 1.0, 1.6, alpha]
    fr = [(0.0, curve(0)), (0.8, curve(0))]
    for k, a in enumerate(stages[1:], 1):
        fr.append((0.8 + 1.0 * k, curve(a)))
    fr += [(7.6, curve(alpha)), (8.6, curve(alpha))]
    whole = (f"<path d='{curve(alpha)}' fill='none' stroke='{BLUE}' stroke-width='3'>"
             + tl.anim("d", curve(alpha), fr) + tl.anim("opacity", 0.3, [(0, 1), (8.4, 1), (9.0, 0.3)]) + "</path>")
    b.append(whole)
    # alpha readout
    for k, a in enumerate(stages):
        t_on = 0.8 + 1.0 * k
        t_off = 0.8 + 1.0 * (k + 1) if k < len(stages) - 1 else 8.6
        b.append(f"<text x='{RX}' y='{P(0, 0.75)[1]}' font-size='20' fill='{BLUE}' font-weight='bold' "
                 f"opacity='0'>α = {a:.2f}{tl.caption(t_on, t_off)}</text>")  # fmt: skip
    b.append(text(RX, P(0, 0.25)[1], "convex ✓", 18, GREEN, extra=tl.caption(5.8, 8.6)))
    gap_full = alpha / 8 * (hi - lo) ** 2
    b.append(text(RX, P(0, -0.35)[1], f"max gap αw²/8 = {gap_full:.2f}", 16, BLUE, extra=tl.show(6.4)))
    # split: two halves with their own alpha
    mid = 0.5 * (lo + hi)
    a1 = max(0.0, 2.25 * max(np.sin(1.5 * z) for z in np.linspace(lo, mid, 400)))
    a2 = max(0.0, 2.25 * max(np.sin(1.5 * z) for z in np.linspace(mid, hi, 400)))
    gap_half = max(a1, a2) / 8 * (mid - lo) ** 2
    halves = (f"<g opacity='1'>{tl.show(9.2)}"
              f"<path d='{curve(a1, lo, mid, 81)}' fill='none' stroke='{ORANGE}' stroke-width='3'/>"
              f"<path d='{curve(a2, mid, hi, 81)}' fill='none' stroke='{ORANGE}' stroke-width='3'/>"
              f"<line x1='{P(mid, 0)[0]}' y1='{P(0, 2.0)[1]}' x2='{P(mid, 0)[0]}' y2='{P(0, -4.6)[1]}' "
              f"stroke='{GREY}' stroke-dasharray='5 4'/>"
              f"{text(RX, P(0, -1.0)[1], f'after one split: {gap_half:.2f}', 16, ORANGE)}"
              f"{text(RX, P(0, -1.5)[1], '4× smaller: gap ∝ width²', 16, ORANGE)}</g>")  # fmt: skip
    b.append(halves)
    caps = captions(tl, [
        (0.2, 5.7, "1 · raise α: the added quadratic is ≤ 0 on the box, so the curve stays below f"),
        (5.8, 9.0, "2 · at α = −min f″ the curvature is nonnegative everywhere: a convex underestimator"),
        (9.1, T - HOLD - 0.5, "3 · on half boxes the gap is 4× smaller: αBB takes over deep in the tree"),
    ], "αBB: whole functions use an interval Hessian; α per variable by scaled Gershgorin")  # fmt: skip
    return svg("".join(b) + caps, "alphaBB animation")


if __name__ == "__main__":
    for name, fn in [("fbbt", fbbt), ("dbbt", dbbt), ("obbt", obbt), ("alphabb", alphabb)]:
        (HERE / f"{name}.svg").write_text(fn())
        print("wrote", HERE / f"{name}.svg")
