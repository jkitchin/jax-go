"""Permutation symmetry of polynomial problems over integer variables, and lex-leader pruning.

``automorphisms`` finds permutations ``p`` of the integer variables that map the problem onto
itself: with ``y = x[p]`` (``y_i = x_{p[i]}``) the objective polynomial is unchanged and the
inequality (equality) rows are permuted among themselves, as exact polynomials. Every element
it returns is checked against the polynomials, so a missed symmetry costs only pruning, never
correctness. The search is the individualize-and-refine scheme of graph automorphism tools
(colour refinement on the variable / term / row graph of the polynomials), building one
transversal per level of a stabilizer chain; the group is then listed as products of the
transversals, up to a cap.

``lex_prune`` uses a list of such elements on boxes: a box is pruned when some element ``p``
maps every point of it to a point that is lexicographically larger (``x[p] >lex x``). The
lex-largest point of every orbit is never pruned that way, and it has the same objective value
and feasibility as every point of its orbit, so a branch and bound that drops pruned boxes keeps
an optimal point and a valid lower bound (Margot, "Symmetry in integer linear programming",
2010). Any subset of the group is valid, which is why the cap and the root-box filter
(``bnb``: elements must keep the variable bounds) are safe.
"""

import time

import numpy as np

MAX_ELEMENTS = 20000  # group elements listed (lex_prune uses the first MAX_PRUNE)
MAX_ENTRIES = 4_000_000  # elements * variables kept
MAX_TERMS = 20000  # monomials over all rows; larger problems are not searched
SEARCH_BUDGET = 20000  # refinement calls in the whole search
SEARCH_SECONDS = 1.5  # wall-clock cap on the search (bnb passes less for short time limits)
MAX_PRUNE = 500  # elements lex_prune uses per call (its cost is linear in them; any subset is valid)


def _clean(p):
    return {k: v for k, v in p.items() if v != 0.0}


def _apply(p, perm):
    """The polynomial ``p(x[perm])`` as a dict (monomials as sorted tuples)."""
    return {tuple(sorted(int(perm[i]) for i in k)): v for k, v in p.items()}


def _key(p):
    return tuple(sorted(p.items()))


def is_automorphism(perm, f, gs, hs):
    """Whether ``x -> x[perm]`` keeps ``f`` and permutes the rows of ``gs`` and of ``hs``."""
    if _apply(f, perm) != f:
        return False
    for rows in (gs, hs):
        if sorted(_key(_apply(p, perm)) for p in rows) != sorted(_key(p) for p in rows):
            return False
    return True


class _Swaps:
    """``is_automorphism`` for the transposition of two variables, looking only at the monomials
    and rows that hold one of them (the others map to themselves)."""

    def __init__(self, f, gs, hs):
        self.f = f
        self.fam = (gs, hs)
        self.f_at, self.rows_at = {}, ({}, {})
        for k in f:
            for i in set(k):
                self.f_at.setdefault(i, []).append(k)
        for at, rows in zip(self.rows_at, self.fam):
            for r, p in enumerate(rows):
                for i in {i for k in p for i in k}:
                    at.setdefault(i, []).append(r)

    def __call__(self, a, b):
        sw = {a: b, b: a}

        def mono(k):
            return tuple(sorted(sw.get(i, i) for i in k))

        for k in self.f_at.get(a, []) + self.f_at.get(b, []):
            if self.f.get(mono(k)) != self.f[k]:
                return False
        for at, rows in zip(self.rows_at, self.fam):
            touched = set(at.get(a, [])) | set(at.get(b, []))
            img = sorted(tuple(sorted((mono(k), v) for k, v in rows[r].items())) for r in touched)
            if img != sorted(_key(rows[r]) for r in touched):
                return False
        return True


class _Graph:
    """Variables 0..n-1, then one node per row, then one per term; edges labelled by exponent."""

    def __init__(self, n, f, gs, hs, ints):
        rows = [("f", f)] + [("g", p) for p in gs] + [("h", p) for p in hs]
        R = len(rows)
        labels = []  # initial colour keys
        for i in range(n):
            labels.append(("int",) if i in ints else ("cont", i))  # continuous variables stay put
        adj = [[] for _ in range(n + R)]
        for r, (kind, p) in enumerate(rows):
            labels.append(("row", kind, p.get((), 0.0) if kind != "f" else r))
        for r, (kind, p) in enumerate(rows):
            for k, v in p.items():
                if not k:
                    continue
                t = len(labels)
                labels.append(("term", float(v)))
                adj.append([])
                adj[t].append((0, n + r))
                adj[n + r].append((0, t))
                for i in set(k):
                    e = k.count(i)
                    adj[t].append((e, i))
                    adj[i].append((e, t))
        keys = sorted(set(labels), key=repr)
        idx = {k: c for c, k in enumerate(keys)}
        self.n = n
        self.c0 = np.asarray([idx[k] for k in labels], np.int64)
        src = np.asarray([u for u in range(len(adj)) for _ in adj[u]], np.int64)
        self.src = src
        self.dst = np.asarray([v for u in range(len(adj)) for _, v in adj[u]], np.int64)
        self.lab = np.asarray([e for u in range(len(adj)) for e, _ in adj[u]], np.uint64)
        self.size = len(adj)


def _mix(z):
    """splitmix64 finalizer (uint64 arrays, wrapping)."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _refine(G, cols):
    """Joint colour refinement of several colourings (shared colour names). Returns the refined
    colourings, or None when their colour histograms differ (no automorphism matches them).
    A node's new colour is a 64-bit hash of its colour and of the multiset of (edge label,
    neighbour colour); a hash collision can only merge colours, which the final check of each
    candidate permutation (``is_automorphism``) catches."""
    cols = [np.asarray(c, np.int64) for c in cols]
    k = len(cols)
    ncol = len(np.unique(np.concatenate(cols)))
    with np.errstate(over="ignore"):
        while True:
            C = np.stack(cols)  # (k, N)
            h = _mix(G.lab * np.uint64(0x9E3779B97F4A7C15) + _mix(C[:, G.dst].astype(np.uint64) + np.uint64(1)))
            S = np.zeros(C.shape, np.uint64)
            for r in range(k):
                np.add.at(S[r], G.src, h[r])
            key = _mix(C.reshape(-1).astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93) + S.reshape(-1))
            uniq, inv = np.unique(key, return_inverse=True)
            new = inv.reshape(k, -1).astype(np.int64)
            m = len(uniq)
            hist = [np.bincount(c, minlength=m) for c in new]
            if any(not np.array_equal(hist[0], q) for q in hist[1:]):
                return None
            cols = list(new)
            if m == ncol:
                return cols
            ncol = m


def automorphisms(n, f, gs, hs, ints, max_elements=MAX_ELEMENTS, budget=SEARCH_BUDGET, seconds=SEARCH_SECONDS,
                  deadline=None, clock=time.perf_counter):
    """Non-identity symmetries of the problem as an ``(S, n)`` int array (rows ``p``: ``x[p]`` is
    feasible with the same objective value whenever ``x`` is), possibly a subset of the group.
    Only integer variables (``ints``) move. Empty when none is found.

    The search stops after ``budget`` refinement calls or once ``clock()`` passes ``deadline``
    (default: ``seconds`` from the start); out of time it returns no elements at all (no
    symmetry handling), so a slow search costs at most that much setup time."""
    deadline = clock() + seconds if deadline is None else deadline
    empty = np.zeros((0, n), np.int64)
    f, gs, hs = _clean(f), [_clean(p) for p in gs], [_clean(p) for p in hs]
    ints = set(int(i) for i in ints)
    if len(ints) < 2 or sum(len(p) for p in [f, *gs, *hs]) > MAX_TERMS:
        return empty
    G = _Graph(n, f, gs, hs, ints)
    swap_ok = _Swaps(f, gs, hs)
    left = [budget]
    late = [False]

    def out_of_time():
        if not late[0] and clock() > deadline:
            late[0] = True
            left[0] = 0
        return late[0]

    def indiv(c, u):
        c = c.copy()
        c[u] = c.max() + 1
        return c

    def extend(L, R):
        """An automorphism mapping the (refined) colouring ``L`` onto ``R``, or None."""
        if left[0] <= 0:
            return None
        cv = L[:n]
        counts = np.bincount(cv)
        multi = np.flatnonzero(counts[cv] > 1)
        if not multi.size:
            perm = np.empty(n, np.int64)
            where = {int(R[w]): w for w in range(n)}
            for u in range(n):
                perm[u] = where[int(L[u])]
            return perm if is_automorphism(perm, f, gs, hs) else None
        u = int(multi[0])
        Lu = indiv(L, u)
        for w in np.flatnonzero(R[:n] == L[u]):
            left[0] -= 1
            if out_of_time():
                return None
            out = _refine(G, [Lu, indiv(R, int(w))])
            if out is not None:
                p = extend(*out)
                if p is not None:
                    return p
            if left[0] <= 0:
                return None
        return None

    base = _refine(G, [G.c0])
    if base is None:
        return empty
    C = base[0]
    transversals = []
    while left[0] > 0:
        cv = C[:n]
        counts = np.bincount(cv)
        multi = np.flatnonzero(counts[cv] > 1)
        if not multi.size:
            break
        b = int(multi[0])
        Lb = indiv(C, b)
        U = []
        for t in np.flatnonzero(cv == cv[b]):
            if t == b:
                continue
            if out_of_time():
                break
            if swap_ok(b, int(t)):  # the transposition (b t): in the stabilizer of the earlier base
                p = np.arange(n)  # points (t shares b's cell, which they are not in) and maps b to t
                p[b], p[t] = t, b
                U.append(p)
                continue
            left[0] -= 1
            out = _refine(G, [Lb, indiv(C, int(t))])
            if out is not None:
                p = extend(*out)
                if p is not None:
                    U.append(p)
            if left[0] <= 0:
                break
        if U:
            transversals.append(U)
        nxt = _refine(G, [Lb])
        if nxt is None:
            break
        C = nxt[0]
    if late[0] or not transversals:
        return empty
    cap = max(1, min(max_elements, MAX_ENTRIES // max(n, 1)))
    # products u_1 u_2 ... u_L (each u_l the identity or an element of transversal l): composing
    # x -> x[a] then x -> x[b] gives x[a][b] = x[a[b]], again a symmetry. Only the rows that fit
    # under the cap are multiplied (E[:k, U] has k |U| rows), so memory stays O(cap n).
    E = np.arange(n)[None]
    for U in transversals:
        U = np.asarray(U, np.int64)
        room = cap + 1 - len(E)
        if room <= 0:
            break
        k = -(-room // len(U))
        E = np.concatenate([E, E[:k, U].reshape(-1, n)[:room]])
    # the transversal elements (a generating set) first, then the other products, so that a
    # prefix (``MAX_PRUNE`` in lex_prune) holds the generators
    E = np.concatenate([np.concatenate([np.asarray(U, np.int64) for U in transversals]), E])
    _, first = np.unique(E, axis=0, return_index=True)
    E = E[np.sort(first)]
    E = E[np.any(E != np.arange(n), axis=1)][:cap]
    return E


def lex_prune(lo, hi, P, max_elements=MAX_PRUNE):
    """Boxes (rows of ``lo``, ``hi``) that hold no lex-leader: some ``p`` among the first
    ``max_elements`` rows of ``P`` has ``x[p] >lex x`` for every ``x`` in the box (lex order over
    ``0..n-1``). Using fewer elements prunes less but stays valid (any subset of the group is)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    B, n = lo.shape
    out = np.zeros(B, bool)
    if not len(P) or not B:
        return out
    P = np.asarray(P, np.int64)[:max_elements]
    bi, gi = np.repeat(np.arange(B), len(P)), np.tile(np.arange(len(P)), B)  # alive (box, element) pairs
    for i in range(n):
        if not bi.size:
            break
        j = P[gi, i]
        same = j == i
        xl, xh, yl, yh = lo[bi, i], hi[bi, i], lo[bi, j], hi[bi, j]
        bigger = ~same & (yl > xh)  # x[p]_i > x_i on the whole box, all earlier positions equal
        out[bi[bigger]] = True
        equal = same | ((xl == xh) & (yl == yh) & (xl == yl))
        keep = equal & ~out[bi]
        bi, gi = bi[keep], gi[keep]
    return out
