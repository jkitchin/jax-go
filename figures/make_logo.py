"""Draw figures/logo.svg: a rugged f (purple), alphaBB underestimators on branch-and-bound
boxes (teal), the certified lower bound and the global minimum (green).

The curves are computed, not hand-drawn: each teal arc is f + alpha (x-a)(x-b) with
alpha >= max(0, -min f''/2) on its box, so it is a valid convex underestimator.
Run: python figures/make_logo.py
"""
import pathlib

import numpy as np
f  = lambda x: 0.6*np.sin(11*x+0.9) + 1.3*(x-0.68)**2
d2 = lambda x: -0.6*121*np.sin(11*x+0.9) + 2.6
X0, X1 = 0.0, 1.25
# icon plotting area
PX0, PX1, PY0, PY1 = 34, 226, 40, 190
xs = np.linspace(X0, X1, 600); fs = f(xs)
cuts = list(np.linspace(0,1.25,6))
rel = []
for a, b in zip(cuts[:-1], cuts[1:]):
    xx = np.linspace(a, b, 200)
    al = max(0.0, -d2(xx).min()/2)
    L = f(xx) + al*(xx-a)*(xx-b)
    rel.append((xx, L))
lo = min(fs.min(), min(L.min() for _, L in rel)); hi = fs.max()
pad = 0.06*(hi-lo)
sx = lambda x: PX0 + (x-X0)/(X1-X0)*(PX1-PX0)
sy = lambda y: PY1 - (y-(lo-pad))/(hi-lo+2*pad)*(PY1-PY0)
path = lambda x, y: "M" + " L".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, y))
xm = xs[fs.argmin()]; ym = fs.min()
parts = []
for (xx, L) in rel:
    fx = f(xx)
    poly = path(xx, fx) + " L" + " L".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xx[::-1], L[::-1])) + " Z"
    parts.append(f'<path d="{poly}" fill="url(#gap)" opacity="0.55"/>')
for c in cuts[1:-1]:
    parts.append(f'<line x1="{sx(c):.1f}" y1="{PY0-6}" x2="{sx(c):.1f}" y2="{PY1+4}" stroke="#8b93a7" stroke-width="1.2" stroke-dasharray="3 4" opacity="0.7"/>')
for xx, L in rel:
    parts.append(f'<path d="{path(xx, L)}" fill="none" stroke="#26a69a" stroke-width="3.2" stroke-linecap="round"/>')
parts.append(f'<path d="{path(xs, fs)}" fill="none" stroke="#b07cff" stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round"/>')
# certified bound: lowest relaxation minimum
lb = min(L.min() for _, L in rel)
parts.append(f'<line x1="{PX0}" y1="{sy(lb):.1f}" x2="{PX1}" y2="{sy(lb):.1f}" stroke="#7cc644" stroke-width="1.5" stroke-dasharray="1 4" stroke-linecap="round"/>')
parts.append(f'<circle cx="{sx(xm):.1f}" cy="{sy(ym):.1f}" r="11" fill="#7cc644" opacity="0.25"/>')
parts.append(f'<circle cx="{sx(xm):.1f}" cy="{sy(ym):.1f}" r="6" fill="#7cc644" stroke="#0f1420" stroke-width="2"/>')
icon = "\n    ".join(parts)
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 230" width="700" height="230" role="img" aria-labelledby="t d">
  <title id="t">jaxglobal</title>
  <desc id="d">A rugged objective (purple) over boxes split by branch and bound, each with a convex underestimator (teal); the global minimum (green) sits above the certified lower bound.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#151a28"/><stop offset="1" stop-color="#0f1420"/>
    </linearGradient>
    <linearGradient id="gap" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#b07cff" stop-opacity="0.35"/><stop offset="1" stop-color="#26a69a" stop-opacity="0.45"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="700" height="230" rx="28" fill="url(#bg)"/>
  <g>
    {icon}
  </g>
  <g font-family="'JetBrains Mono','Fira Code',Menlo,Consolas,monospace" font-weight="700">
    <text x="258" y="122" font-size="64" fill="#e8ebf2"><tspan fill="#26a69a">j</tspan><tspan fill="#b07cff">a</tspan><tspan fill="#7cc644">x</tspan>global</text>
  </g>
  <text x="261" y="160" font-family="-apple-system,'Segoe UI',Helvetica,Arial,sans-serif" font-size="18" fill="#9aa3b8" letter-spacing="0.2">certified global optima of plain JAX functions</text>
</svg>
'''
(pathlib.Path(__file__).parent / "logo.svg").write_text(svg)
