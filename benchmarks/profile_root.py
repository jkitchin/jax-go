"""Where does the wall time of a MINLPLib solve go before and at the root node?

Runs ``minimize`` on .nl instances, one child process per (instance, mode),
one after another, and splits the wall time of ``minimize`` into stages. The
solver is not changed: the script wraps functions with timers, and
``bnb._mark`` (a no-op unless ``bnb._PROFILE`` is set) tells it when each
setup / root stage of ``minimize`` begins.

Modes:
  cold   ``JAXGLOBAL_CACHE=0``: no kernel cache, no XLA compilation cache
  prime  an empty cache directory (fills it: cold plus the export cost)
  warm   the directory ``prime`` filled

What is recorded per run (JSON from the child, rows of the CSV):
  * main-thread stages (``bnb._mark``), which partition ``minimize``'s wall time;
  * on the main thread, inside them: waiting for a kernel executable, running
    it, ``bound_domain`` and its parts, ``make_local``, each local solve, the
    structure analysis, kcache fingerprints, and every JAX trace / lowering /
    XLA compile (``jax.monitoring`` spans);
  * on the background threads (tracer, compiler pool, domain): trace + lowering
    and XLA compile of each kernel, with what they overlapped;
  * which paths were taken: ``use_interpreter`` / ``use_relax_interpreter``
    decisions with their row counts, the quadratic path, alphaBB's
    ``jacfwd(jacfwd)`` fallback, the scalarized rows, nodes and status.

Usage:
  python benchmarks/profile_root.py --names ex6_2_6,ramsey --time-limit 120 \
      --out benchmarks/results/scipgap_profile.csv [--nl-dir DIR] [--modes cold,prime,warm]

Diagnostic variants (the mode is recorded as ``<mode><label>``):
  --opts '{"alphabb": false}' --label +noalphabb        extra ``minimize`` options
  --set '{"relax_interp.RULE": "rows", "relax_interp.MIN_ROWS": 0}' --label +interp  module constants

Per-node kernel ablation (plan A3): ``--ablate`` runs the fixed variant list ``ABLATE`` cold and
prints a table of compile s (node bound kernel), kernel run s per call, nodes/s and the bound at the
limit, medians over ``--repeats``. Each run holds the shared benchmark lock and waits for a 1-min
load below ``--max-load`` (10); the load is recorded, and a run started after ``--max-wait`` s of
waiting is flagged ``timeout`` (``rose``: the load passed the gate but was above it again once the
lock was taken; the lock is never held while waiting for the load):
  python benchmarks/profile_root.py --ablate --names hadamard_5,ex6_2_6 --time-limit 60 --repeats 3 \\
      --out benchmarks/results/scipgap_a3.csv [--variants base,+opsrule,+interp]

A background compile still running when ``minimize`` returns is recorded up to
the return and marked unfinished.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import pathlib
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = "PROFILE-RESULT "
FIELDS = ["instance", "mode", "load_1_5_15", "category", "item", "seconds", "note"]


# ---------------------------------------------------------------- child


def child(name, nl_dir, tl):
    t_import = time.time()
    sys.path.insert(0, str(ROOT))
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpy as np

    import jaxglobal
    from jaxglobal import alphabb, bnb, domain, fbbt_interp, kcache, relax_interp
    from jaxglobal.nl import read_nl

    assert pathlib.Path(jaxglobal.__file__).resolve().is_relative_to(ROOT), jaxglobal.__file__
    spans, marks, paths = [], [], []
    lock = threading.Lock()

    def span(kind, item, t0, t1, note=""):
        with lock:
            spans.append(dict(kind=kind, item=item, thread=threading.current_thread().name, t0=t0, t1=t1, note=note))

    running = {}  # background spans still in flight, recorded as unfinished at the end

    def timed(kind, item, fn, note=None):
        def wrapper(*a, **k):
            t = time.time()
            key = object()
            with lock:
                running[key] = (kind, item, threading.current_thread().name, t)
            try:
                return fn(*a, **k)
            finally:
                with lock:
                    running.pop(key, None)
                span(kind, item, t, time.time(), note(a, k) if note else "")

        return wrapper

    # every JAX trace / lowering / backend compile, on whatever thread it runs
    short = {"/jax/core/compile/jaxpr_trace_duration": "jax_trace",
             "/jax/core/compile/jaxpr_to_mlir_module_duration": "jax_lower",
             "/jax/core/compile/backend_compile_duration": "jax_compile"}  # fmt: skip

    def on_span(event, t0, t1, **kw):
        if event in short:
            span(short[event], kw.get("fun_name", ""), t0, t1)

    jax.monitoring.register_event_time_span_listener(on_span)
    bnb._PROFILE = lambda stage: marks.append((stage, time.time()))

    # setup work on the main thread
    bnb.bound_domain = timed("fn", "bound_domain", bnb.bound_domain)
    domain._feasible_points = timed("fn", "domain_feasible_points", domain._feasible_points)
    domain.used_variables = timed("fn", "domain_used_variables", domain.used_variables)
    bnb.crown_auto = timed("fn", "crown_auto", bnb.crown_auto)
    bnb.nonlinear_variables = timed("fn", "nonlinear_variables_hessian", bnb.nonlinear_variables)
    for meth in ("structure", "quadratic", "nonlinear", "fingerprint"):
        setattr(bnb._Kernels, meth, timed("fn", f"K.{meth}", getattr(bnb._Kernels, meth)))
    kcache.load = timed("fn", "kcache_load", kcache.load)
    kcache.save = timed("fn", "kcache_save", kcache.save)
    _export = jax.export.export

    def export(fn, *a, **k):
        return timed("fn", "kcache_export_trace_lower", _export(fn, *a, **k))

    jax.export.export = export

    n_local = [0]

    def make_local(*a, **k):
        t = time.time()
        solve = bnb_make_local(*a, **k)
        span("fn", "make_local", t, time.time())

        def timed_solve(X0, *b, **kw):
            n_local[0] += 1
            return timed("fn", "local_solve", solve, lambda a_, k_: f"#{n_local[0]} starts={len(X0)}")(X0, *b, **kw)

        return timed_solve

    bnb_make_local = bnb.make_local
    bnb.make_local = make_local

    # the kernels: trace + lowering on the tracer thread, compile on the compiler pool
    class TimedLowered:
        def __init__(self, lowered, item):
            self.lowered, self.item = lowered, item

        def compile(self, *a, **k):
            return timed("bg", f"compile:{self.item}", self.lowered.compile)(*a, **k)

    _lower = bnb._Precompiled._lower

    def lower(self, args, fingerprint):
        item = f"{self.name}[b={args[0].shape[0]}]"
        return TimedLowered(timed("bg", f"trace_lower:{item}", _lower)(self, args, fingerprint), item)

    bnb._Precompiled._lower = lower
    _call = bnb._Precompiled.__call__

    def call(self, *args, deadline=None):
        fut = self.ready.get(self._key(args))
        if fut is not None:  # wait here (same deadline), so the wait and the run are timed apart
            t = time.time()
            try:
                wait = None if deadline is None or not np.isfinite(deadline) else max(deadline - time.perf_counter(), 0)
                fut.result(timeout=wait).result(timeout=wait)
            except Exception:
                pass
            span("fn", f"wait:{self.name}", t, time.time())
        t = time.time()
        out = _call(self, *args, deadline=deadline)
        jax.block_until_ready(out)
        span("fn", f"run:{self.name}", t, time.time(), f"boxes={args[0].shape[0]}")
        return out

    bnb._Precompiled.__call__ = call

    class TimedJit:  # the dive kernel: trace().lower().compile() on the background threads
        def __init__(self, obj, item, stage=""):
            self.obj, self.item, self.stage = obj, item, stage

        def __call__(self, *a, **k):
            return self.obj(*a, **k)

        def trace(self, *a, **k):
            return TimedJit(timed("bg", f"trace:{self.item}", self.obj.trace)(*a, **k), self.item, "traced")

        def lower(self, *a, **k):
            return TimedJit(timed("bg", f"lower:{self.item}", self.obj.lower)(*a, **k), self.item, "lowered")

        def compile(self, *a, **k):
            return timed("bg", f"compile:{self.item}", self.obj.compile)(*a, **k)

    _init = bnb._Kernels.__init__

    def init(self, *a, **k):
        t = time.time()
        _init(self, *a, **k)
        span("fn", "K.__init__", t, time.time())

    bnb._Kernels.__init__ = init

    # which paths were taken
    def decision(mod):
        orig = mod.use_interpreter

        def use_interpreter(tape, n, *a, **k):
            r = orig(tape, n, *a, **k)
            rows, _ = fbbt_interp.tape_size(tape)
            f1, f2 = sys._getframe(1), sys._getframe(2)
            who = f"{f2.f_code.co_name}>{f1.f_code.co_name}" if f1.f_code.co_name == "use_relax_interpreter" else \
                f"{f1.f_globals.get('__name__', '').split('.')[-1]}.{f1.f_code.co_name}"  # fmt: skip
            with lock:
                paths.append(dict(what="use_interpreter", who=who, args=list(a) + sorted(k.items()), rows=rows,
                                  result=bool(r)))  # fmt: skip
            return r

        mod.use_interpreter = use_interpreter

    decision(fbbt_interp)
    decision(relax_interp)
    _relax_rule = relax_interp.use_relax_interpreter

    def use_relax_interpreter(tape, n):
        r = _relax_rule(tape, n)
        with lock:
            paths.append(dict(what="use_relax_interpreter", rule=relax_interp.RULE, n=n,
                              rows=fbbt_interp.tape_size(tape)[0], ops=relax_interp.kernel_ops(tape, n), result=bool(r)))
        return r

    relax_interp.use_relax_interpreter = use_relax_interpreter
    _taylor2 = relax_interp.make_taylor2

    def make_taylor2(tape, n, ictx, d):
        run = _taylor2(tape, n, ictx, d)
        size = None if run is None else alphabb.taylor2_bytes(int(run.program.const_val.size), d)
        with lock:
            paths.append(dict(what="taylor2", who=sys._getframe(1).f_code.co_name, bytes=size,
                              budget=alphabb.TAYLOR_BYTES, over=None if size is None else size > alphabb.TAYLOR_BYTES))  # fmt: skip
        return run

    relax_interp.make_taylor2 = make_taylor2
    _evmc = alphabb.eval_jaxpr_mc

    def eval_jaxpr_mc(*a, **k):
        if sys._getframe(1).f_code.co_name == "hess":  # alphabb's traced interval Hessian
            with lock:
                paths.append(dict(what="alphabb_jacfwd_fallback", eqns=len(a[0].eqns)))
        return _evmc(*a, **k)

    alphabb.eval_jaxpr_mc = eval_jaxpr_mc

    t_read = time.time()
    M = read_nl(pathlib.Path(nl_dir) / f"{name}.nl", integers="mask")
    # diagnostic variants (--opts / --set): extra minimize options and module constants
    opts = json.loads(os.environ.get("JAXGLOBAL_PROFILE_OPTS", "{}"))
    for key, val in json.loads(os.environ.get("JAXGLOBAL_PROFILE_SET", "{}")).items():
        mod, attr = key.rsplit(".", 1)
        setattr(importlib.import_module(f"jaxglobal.{mod}"), attr, val)
    t_min = time.time()
    r = bnb.minimize(M.f, M.lo, M.hi, ineq=M.ineq, eq=M.eq, integer=M.integer, abs_tol=1e-6, rel_tol=1e-4,
                     feas_tol=1e-6, time_limit=tl, **opts)  # fmt: skip
    t_end = time.time()
    with lock:
        for kind, item, thread, t in running.values():
            spans.append(dict(kind=kind, item=item, thread=thread, t0=t, t1=t_end, note="unfinished at return"))

    # facts read after the solve (not timed into it)
    from jaxglobal.fbbt import joint_tape

    n = int(np.asarray(M.lo).size)
    info = dict(n=n, n_int=int(np.sum(M.integer)) if M.integer is not None else 0,
                free=int(np.sum(~(np.isfinite(M.lo) & np.isfinite(M.hi)))), status=r.status, nodes=int(r.nodes),
                fun=float(r.fun), lower_bound=float(r.lower_bound), message=r.message)  # fmt: skip
    try:
        with jax.ensure_compile_time_eval():
            tape = joint_tape(M.f, M.ineq, M.eq, np.zeros(n))
        prog = fbbt_interp.scalarize(tape, n)
        info["rows"] = None if prog is None else int(len(prog.op))
    except Exception as e:
        info["rows"] = f"error {e!r}"[:80]
    K = next((v for k, v in bnb._CACHE.items() if k[0] == "kernels"), None)
    if K is not None:
        info["quadratic"] = any(v is not None for v in K._quadratic.values())
        info["structure"] = any(v is not None for v in K._structure.values())
    return dict(name=name, t_import=t_import, t_read=t_read, t_min=t_min, t_end=t_end, marks=marks, spans=spans,
                paths=paths, info=info)  # fmt: skip


# ---------------------------------------------------------------- analysis


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _union(iv):
    tot, end = 0.0, -1e300
    for a, b in sorted(iv):
        if b > end:
            tot += b - max(a, end)
            end = b
    return tot


def analyse(d):
    """Rows (category, item, seconds, note) for one run."""
    rows = []
    t_min, t_end = d["t_min"], d["t_end"]
    wall = t_end - t_min
    rows.append(("total", "minimize_wall", wall, d["info"]["status"]))
    rows.append(("total", "import_jax_jaxglobal", d["t_read"] - d["t_import"], "not in minimize"))
    rows.append(("total", "read_nl", t_min - d["t_read"], "not in minimize"))
    # main-thread stages from the marks
    marks = [("pre_domain", t_min)] + [tuple(m) for m in d["marks"]] + [("end", t_end)]
    stages = {}
    for (s, a), (_, b) in zip(marks, marks[1:]):
        stages.setdefault(s, []).append((a, b))
    for s, iv in stages.items():
        rows.append(("stage", s, sum(b - a for a, b in iv), ""))
    main = [x for x in d["spans"] if x["thread"] == "MainThread"]
    bg = [x for x in d["spans"] if x["thread"] != "MainThread"]
    # main-thread details, per stage
    for s, iv in stages.items():
        det = {}
        for kind in ("jax_trace", "jax_lower", "jax_compile"):  # nested events counted once
            sec = _union([(max(x["t0"], a), min(x["t1"], b)) for x in main if x["kind"] == kind
                          for a, b in iv if _overlap(x["t0"], x["t1"], a, b) > 0])  # fmt: skip
            if sec > 0:
                det[kind] = [sec, 0, []]
        for x in main:
            if x["kind"] == "fn":
                o = sum(_overlap(x["t0"], x["t1"], a, b) for a, b in iv)
                if o > 0:
                    key = x["item"]
                    det.setdefault(key, [0.0, 0, []])
                    det[key][0] += o
                    det[key][1] += 1
                    if x["note"]:
                        det[key][2].append(x["note"])
        for k, (sec, cnt, notes) in sorted(det.items(), key=lambda kv: -kv[1][0]):
            if sec >= 0.005:
                rows.append(("main_in_" + s, k, sec, f"calls={cnt}" + (" " + ";".join(notes[:3]) if notes else "")))
    # union of all JAX trace / lowering / compile on the main thread (nested events counted once)
    for kind in ("jax_trace", "jax_lower", "jax_compile"):
        rows.append(("main_jax", kind, _union([(x["t0"], x["t1"]) for x in main if x["kind"] == kind]), "union"))
    # background work, with the main-thread stages it overlapped
    groups = {}
    for x in bg:
        key = (x["item"] if x["kind"] in ("bg", "fn") else f"{x['kind']}@{x['thread'].split('_')[0]}")
        groups.setdefault(key, []).append(x)
    for key, xs in groups.items():
        iv = [(x["t0"], x["t1"]) for x in xs]
        sec = _union(iv) if key.startswith("jax_") else sum(b - a for a, b in iv)
        ov = {s: sum(_overlap(a, b, p, q) for a, b in iv for p, q in siv) for s, siv in stages.items()}
        ov = ", ".join(f"{s} {v:.1f}" for s, v in sorted(ov.items(), key=lambda kv: -kv[1]) if v >= 0.05)
        end = max(b for _, b in iv) - t_min
        unf = "; UNFINISHED when minimize returned" if any(x.get("note") == "unfinished at return" for x in xs) else ""
        rows.append(("background", key, sec, f"ends at {end:.1f}s; overlaps {ov}{unf}"))
    # paths taken
    seen = set()
    for p in d["paths"]:
        k = json.dumps(p, sort_keys=True)
        if k not in seen:
            seen.add(k)
            rows.append(("path", p["what"], "", " ".join(f"{a}={b}" for a, b in p.items() if a != "what")))
    for k, v in d["info"].items():
        rows.append(("info", k, "", v))
    return rows, wall, stages


def buckets(d):
    """The wall time of ``minimize`` in a few buckets that add up to it (see the module docstring)."""
    rows, wall, stages = analyse(d)
    main = [x for x in d["spans"] if x["thread"] == "MainThread"]
    bg = [x for x in d["spans"] if x["thread"] != "MainThread"]

    def in_stage(names, pred):
        return sum(_overlap(x["t0"], x["t1"], a, b) for s in names for a, b in stages.get(s, []) for x in main if pred(x))

    def stage(*names):
        return sum(b - a for s in names for a, b in stages.get(s, []))

    out = {}
    # waiting for a kernel: split by what the background was doing (its trace+lower, else compile)
    waits = [x for x in main if x["item"].startswith("wait:")]
    tr = [(x["t0"], x["t1"]) for x in bg if x["item"].startswith(("trace_lower:", "trace:", "lower:"))]
    co = [(x["t0"], x["t1"]) for x in bg if x["item"].startswith("compile:")]
    w_trace = w_comp = w_other = 0.0
    for w in waits:
        for t in range(int((w["t1"] - w["t0"]) / 0.01) + 1):  # 10 ms steps
            p = w["t0"] + (t + 0.5) * 0.01
            if p > w["t1"]:
                break
            dt = min(0.01, w["t1"] - (w["t0"] + t * 0.01))
            if any(a <= p < b for a, b in co):
                w_comp += dt
            elif any(a <= p < b for a, b in tr):
                w_trace += dt
            else:
                w_other += dt
    out["kernel wait: trace+lower"] = w_trace
    out["kernel wait: XLA compile"] = w_comp
    out["kernel wait: other"] = w_other
    out["bound_domain"] = stage("bound_domain")
    out["structure/nonlinear/quadratic"] = stage("nonlinear", "quadratic")
    out["make_local"] = stage("make_local")
    run_bound = in_stage(["root_bound"], lambda x: x["item"] == "run:bound")
    out["root bound: kernel run (first LP)"] = run_bound
    out["root local solves + incumbent"] = stage("root_incumbent", "root_obbt_incumbent")
    # the dive kernel (``K.dive.wait``) is waited for inside ``dive``: count the time
    # the root dives stage overlaps its trace / compile as kernel wait
    dv_tr = [(max(x["t0"], a), min(x["t1"], b)) for x in bg if x["item"].startswith(("trace_lower:dive", "trace:dive", "lower:dive"))
             for a, b in stages.get("root_dives", []) if _overlap(x["t0"], x["t1"], a, b) > 0]  # fmt: skip
    dv_co = [(max(x["t0"], a), min(x["t1"], b)) for x in bg if x["item"].startswith("compile:dive")
             for a, b in stages.get("root_dives", []) if _overlap(x["t0"], x["t1"], a, b) > 0]  # fmt: skip
    out["kernel wait: trace+lower"] += _union(dv_tr)
    out["kernel wait: XLA compile"] += _union(dv_co)
    out["root dives + enum"] = stage("root_dives", "root_enum") - _union(dv_tr) - _union(dv_co)
    wait_obbt = in_stage(["root_obbt"], lambda x: x["item"] == "wait:obbt")
    out["root OBBT: kernel run"] = stage("root_obbt") - wait_obbt
    wait_probe = in_stage(["probe"], lambda x: x["item"].startswith("wait:"))
    out["probing: run"] = stage("probe") - wait_probe
    out["search"] = stage("search")
    out["setup other (pre_domain, setup, kernels, start_compile)"] = stage(
        "pre_domain", "setup", "kernels", "start_compile", "start_dive_compile", "end")
    # the root_bound stage minus its wait and run
    wait_bound = in_stage(["root_bound"], lambda x: x["item"] == "wait:bound")
    out["root bound: other"] = stage("root_bound") - wait_bound - run_bound
    # (waits inside root_obbt / probe are already in "kernel wait"; take them out of the search part)
    wait_search = in_stage(["search"], lambda x: x["item"].startswith("wait:"))
    out["search"] -= wait_search
    return rows, wall, out


def ablation(d):
    """The numbers of the ``--ablate`` table for one run.

    compile_s: trace + lowering + XLA compile of the node bound kernel (16 boxes), as run on the
    background threads; run_s: seconds per call of that kernel during the search; nodes_s: nodes per
    second of the search stage; bound: the lower bound when ``minimize`` returned.
    """
    _, wall, stages = analyse(d)
    bg = [x for x in d["spans"] if x["thread"] != "MainThread"]
    main = [x for x in d["spans"] if x["thread"] == "MainThread"]
    comp = sum(x["t1"] - x["t0"] for x in bg if x["item"] in ("trace_lower:bound[b=16]", "compile:bound[b=16]"))
    unfinished = any(x.get("note") == "unfinished at return" for x in bg if x["item"].endswith("bound[b=16]"))
    search = stages.get("search", [])
    runs = [x["t1"] - x["t0"] for x in main if x["item"] == "run:bound"
            and any(_overlap(x["t0"], x["t1"], a, b) > 0 for a, b in search)]  # fmt: skip
    t_search = sum(b - a for a, b in search)
    interp = sorted({p["rows"] for p in d["paths"] if p["what"] == "use_relax_interpreter" and p["result"]})
    info = d["info"]
    return dict(wall=wall, compile_s=comp, compile_unfinished=unfinished,
                run_s=sum(runs) / len(runs) if runs else float("nan"), calls=len(runs),
                nodes_s=info["nodes"] / t_search if t_search > 0 else float("nan"), nodes=info["nodes"],
                bound=info["lower_bound"], status=info["status"], interp_rows=interp)  # fmt: skip


# ---------------------------------------------------------------- driver

LOCK = pathlib.Path("/private/tmp/claude-501/jaxglobal-bench.lock")

# the fixed variant list of ``--ablate`` (plan A3): (label, minimize options, module constants)
ABLATE = [
    ("", {}, {}),
    ("+opsrule", {}, {"relax_interp.RULE": "ops"}),  # the candidate op-count rule (relax_interp.MIN_OPS)
    ("+noalphabb", {"alphabb": False}, {}),
    ("+nounivariate", {"univariate": False}, {}),
    ("+nokelley", {"n_kelley": 0}, {}),
    ("+fbbt1", {"fbbt_rounds": 1}, {}),
    ("+noobbt", {"obbt": False}, {}),
    ("+interp", {}, {"relax_interp.RULE": "rows", "relax_interp.MIN_ROWS": 0}),
    ("+straight", {}, {"relax_interp.RULE": "rows", "relax_interp.MIN_ROWS": 10**9}),
    ("+fbbtrows", {}, {"fbbt_interp.SCHEDULE_MIN_ROWS": 0}),
    ("+fbbtsteps", {}, {"fbbt_interp.SCHEDULE_MIN_ROWS": 10**9}),
]


def _load():
    try:
        out = subprocess.run(["uptime"], capture_output=True, text=True).stdout
        return out.strip().split("load average")[-1].lstrip("s:").strip()
    except Exception:
        return ""


def gate(max_load, max_wait, poll=20.0):
    """Wait until the 1-min load is below ``max_load``, at most ``max_wait`` s: (load, "ok" | "timeout")."""
    t = time.time()
    while (ld := os.getloadavg()[0]) >= max_load:
        if time.time() - t >= max_wait:
            return ld, "timeout"
        time.sleep(poll)
    return ld, "ok"


class Lock:
    """The shared benchmark lock of plan §3 (atomic ``mkdir``), held for one run; no-op without a path.

    The owner file carries the pid and a random token; ``__exit__`` removes the lock only if the owner
    file still names this holder, so a lock another process took after a stale-lock cleanup survives.
    """

    def __init__(self, path, what, poll=15.0):
        self.path, self.what, self.poll = pathlib.Path(path) if path else None, what, poll
        self.token = f"pid {os.getpid()} token {uuid.uuid4().hex}"

    def __enter__(self):
        i = 0
        while self.path is not None:
            try:
                self.path.mkdir()
            except FileExistsError:
                if i % 40 == 0:
                    owner = (self.path / "owner").read_text().strip() if (self.path / "owner").exists() else "?"
                    print(f"    waiting for {self.path} ({owner})", flush=True)
                i += 1
                time.sleep(self.poll)
                continue
            try:
                (self.path / "owner").write_text(f"profile_root.py {self.token} {self.what} since {time.ctime()}\n")
            except BaseException:
                shutil.rmtree(self.path, ignore_errors=True)  # never leave an ownerless lock behind
                raise
            break
        return self

    def __exit__(self, *exc):
        if self.path is None:
            return
        try:
            mine = self.token in (self.path / "owner").read_text()
        except OSError:
            mine = False
        if mine:
            shutil.rmtree(self.path, ignore_errors=True)
        else:
            print(f"    {self.path} is no longer ours; left in place", flush=True)


def _exit_on_signal(signum, frame):
    """SIGTERM/SIGHUP -> SystemExit, so ``with Lock(...)`` unwinds and releases the lock."""
    raise SystemExit(128 + signum)


def run_one(a, w, jdir, cache, name, mode, label, opts, sets):
    """One child run: rows into the CSV writer ``w``, the raw JSON into ``jdir``. Returns (d, load, gate) or None."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("JAXGLOBAL_CACHE", "JAX_COMPILATION_CACHE"))}
    env["PYTHONWARNINGS"] = "ignore"
    env["JAXGLOBAL_PROFILE_OPTS"] = json.dumps(opts)
    env["JAXGLOBAL_PROFILE_SET"] = json.dumps(sets)
    if mode == "cold":
        env["JAXGLOBAL_CACHE"] = "0"
    else:
        env["JAXGLOBAL_CACHE_DIR"] = str(cache / name)
    tag = mode + label
    # wait for the load before taking the lock (never hold it while waiting), then record the load
    # under the lock: "rose" if it went back above the gate in between
    outcome = gate(a.max_load, a.max_wait)[1] if a.max_load > 0 else "off"
    with Lock(a.lock, f"{name} {tag}, one run of <= {a.time_limit * 3 + 120:.0f} s"):
        ld = os.getloadavg()[0]
        if outcome == "ok" and ld >= a.max_load:
            outcome = "rose"
        load = _load()
        t = time.time()
        try:
            p = subprocess.run([sys.executable, __file__, "--child", name, a.nl_dir, str(a.time_limit)],
                               capture_output=True, text=True, env=env, timeout=a.time_limit * 3 + 120)  # fmt: skip
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith(TAG)), None)
            err = (p.stderr or p.stdout)[-200:]
        except subprocess.TimeoutExpired:
            line, err = None, "child timed out"
    gnote = f"gate={outcome} load1={ld:.2f}"
    if line is None:
        print(f"{name} {tag}: no result ({err})", flush=True)
        w.writerow(dict(instance=name, mode=tag, load_1_5_15=load, category="error", item="", seconds="",
                        note=f"{gnote} {err}"))  # fmt: skip
        return None
    d = json.loads(line[len(TAG):])
    (jdir / f"{name}-{tag}.json").write_text(line[len(TAG):])
    rows, wall, bk = buckets(d)
    rows.append(("info", "load_gate", "", gnote))
    for cat, item, sec, note in rows:
        w.writerow(dict(instance=name, mode=tag, load_1_5_15=load, category=cat, item=item,
                        seconds=f"{sec:.3f}" if isinstance(sec, float) else sec, note=note))  # fmt: skip
    for item, sec in bk.items():
        w.writerow(dict(instance=name, mode=tag, load_1_5_15=load, category="bucket", item=item,
                        seconds=f"{sec:.3f}", note=""))  # fmt: skip
    print(f"{name:24s} {tag:5s} load {load:18s} ({outcome}) wall {wall:7.1f}s (process {time.time() - t:.0f}s) "
          f"buckets sum {sum(bk.values()):7.1f}s  nodes {d['info']['nodes']} {d['info']['status']}", flush=True)  # fmt: skip
    if not a.ablate:
        for item, sec in sorted(bk.items(), key=lambda kv: -kv[1]):
            if sec >= 0.05:
                print(f"    {item:55s} {sec:7.2f}")
    return d, ld, outcome


def _med(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")


def ablation_table(results):
    """Markdown table of ``--ablate`` results: medians over the repeats of each (instance, variant)."""
    lines = ["| instance | variant | compile s | run s/call | nodes/s | bound at limit | status | relax interp rows | "
             "load (1 min) | gate |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for (name, tag), reps in results.items():
        ab = [ablation(d) for d, _, _ in reps]
        unf = "+" if any(x["compile_unfinished"] for x in ab) else ""
        lines.append(
            f"| {name} | {tag} | {_med([x['compile_s'] for x in ab]):.1f}{unf} | {_med([x['run_s'] for x in ab]):.4f} | "
            f"{_med([x['nodes_s'] for x in ab]):.0f} | {_med([x['bound'] for x in ab]):.6g} | "
            f"{'/'.join(sorted({x['status'] for x in ab}))} | {ab[0]['interp_rows'] or '-'} | "
            f"{', '.join(f'{ld:.1f}' for _, ld, _ in reps)} | {'/'.join(sorted({g for _, _, g in reps}))} |")  # fmt: skip
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="")
    ap.add_argument("--nl-dir", default=str(HERE / "minlplib" / "nl"))
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--modes", default="cold,prime,warm")
    ap.add_argument("--out", default=str(HERE / "results" / "scipgap_profile.csv"))
    ap.add_argument("--json-dir", default="", help="keep each run's raw JSON here (default: a temp dir)")
    ap.add_argument("--opts", default="", help='extra minimize options as JSON, e.g. \'{"alphabb": false}\'')
    ap.add_argument("--set", default="", help='module constants as JSON, e.g. \'{"relax_interp.MIN_ROWS": 0}\'')
    ap.add_argument("--label", default="", help="appended to the mode in the CSV (diagnostic variants)")
    ap.add_argument("--ablate", action="store_true",
                    help="run the fixed variant list ABLATE (cold) and print the ablation table")  # fmt: skip
    ap.add_argument("--variants", default="", help="--ablate: only these labels (comma-separated, 'base' for '')")
    ap.add_argument("--repeats", type=int, default=1, help="runs per (instance, variant); the table shows medians")
    ap.add_argument("--max-load", type=float, default=None,
                    help="wait for a 1-min load below this before each run (default 10 with --ablate, else 0: no gate)")
    ap.add_argument("--max-wait", type=float, default=7200.0, help="s to wait for the load gate, then run flagged")
    ap.add_argument("--lock", default=None, help=f"benchmark lock directory held during each run (default {LOCK} "
                    "with --ablate, else none; a caller that holds the lock itself must not pass it)")
    ap.add_argument("--child", nargs=3, metavar=("NAME", "NL_DIR", "TL"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.max_load is None:
        a.max_load = 10.0 if a.ablate else 0.0
    if a.lock is None:
        a.lock = str(LOCK) if a.ablate else ""
    if a.child:
        print(TAG + json.dumps(child(a.child[0], a.child[1], float(a.child[2])), default=str), flush=True)
        sys.stdout.flush()
        os._exit(0)  # do not wait for a kernel still compiling in the background

    jdir = pathlib.Path(a.json_dir or tempfile.mkdtemp(prefix="jaxglobal-profile-"))
    jdir.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(tempfile.mkdtemp(prefix="jaxglobal-profile-cache-"))
    if a.ablate:
        keep = {"" if v == "base" else v for v in a.variants.split(",")} if a.variants else None
        plan = [("cold", label, opts, sets) for label, opts, sets in ABLATE if keep is None or label in keep]
    else:
        plan = [(mode, a.label, json.loads(a.opts or "{}"), json.loads(a.set or "{}")) for mode in a.modes.split(",")]
    if a.lock:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _exit_on_signal)
    results = {}
    new = not pathlib.Path(a.out).exists()
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        runs = [(name, *step) for name in a.names.split(",") for step in plan for _ in range(a.repeats)]
        for k, (name, mode, label, opts, sets) in enumerate(runs):
            r = run_one(a, w, jdir, cache, name, mode, label, opts, sets)
            fh.flush()
            if a.lock and k + 1 < len(runs):
                time.sleep(45)  # let others waiting for the lock through between runs
            if r is not None:
                results.setdefault((name, mode + label), []).append(r)
    if a.ablate or a.repeats > 1:
        print(ablation_table(results))
    print(f"raw JSON in {jdir}")


if __name__ == "__main__":
    main()
