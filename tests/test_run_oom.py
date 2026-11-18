"""benchmarks/run.py records solver processes killed by a signal (OOM) and their peak RSS."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

import run  # noqa: E402


def test_sigkill_is_oom_with_peak_rss():
    r = run._run([sys.executable, "-c", "import os, signal; x = bytearray(100 << 20); x[::4096] = b'1' * len(x[::4096]); "
                  "os.kill(os.getpid(), signal.SIGKILL)"], 10)  # fmt: skip
    assert r["status"] == "oom" and "SIGKILL" in r["message"]
    assert float(run._RUN_INFO.peak_rss_mb) >= 100


def test_other_signal_is_killed():
    r = run._run([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"], 10)
    assert r["status"] == "killed" and "SIGTERM" in r["message"]


def test_normal_exit_returns_output():
    assert run._run([sys.executable, "-c", "print('hi')"], 10).strip() == "hi"
    assert float(run._RUN_INFO.peak_rss_mb) > 0


def test_memory_limit(monkeypatch):
    monkeypatch.setattr(run, "MEM_LIMIT_MB", 50)
    r = run._run([sys.executable, "-c", "import time; x = bytearray(300 << 20); x[::4096] = b'1' * len(x[::4096]); "
                  "time.sleep(20)"], 10)  # fmt: skip
    assert r["status"] == "oom" and "memory limit" in r["message"]
