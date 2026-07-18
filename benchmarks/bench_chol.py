"""Reference benchmark: pure-NumPy cholupdate vs vendored Cython choldate.

Run from the repo root: uv run python benchmarks/bench_chol.py
Numbers are for reference only; optimization is not a project goal.
"""
import time

import numpy as np

import pycss.subset_selection as ss
from pycss.chol import cholupdate as np_cholupdate

try:
    from choldate import cholupdate as c_cholupdate
except ImportError:
    c_cholupdate = None


def bench_update(fn, p, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((p, p))
    A = M @ M.T + p * np.eye(p)
    R0 = np.ascontiguousarray(np.linalg.cholesky(A).T)
    xs = rng.standard_normal((reps, p))
    start = time.perf_counter()
    for i in range(reps):
        fn(R0.copy(), xs[i].copy())
    return (time.perf_counter() - start) / reps


def bench_swapping(backend, p=100, k=10, seed=0):
    """Time swapping_css with pycss.subset_selection's cholupdate swapped out."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((2 * p, p))
    Sigma = M.T @ M / (2 * p)
    original = ss.cholupdate
    ss.cholupdate = backend
    try:
        start = time.perf_counter()
        ss.swapping_css(Sigma, k, S_init=np.arange(k))
        return time.perf_counter() - start
    finally:
        ss.cholupdate = original


def main():
    print("cholupdate micro-benchmark (mean per call)")
    print(f"{'p':>6} {'numpy (ms)':>12} {'choldate (ms)':>14} {'numpy/choldate':>15}")
    for p in [10, 50, 100, 250, 500, 1000]:
        t_np = bench_update(np_cholupdate, p) * 1e3
        if c_cholupdate is not None:
            t_c = bench_update(c_cholupdate, p) * 1e3
            print(f"{p:>6} {t_np:>12.4f} {t_c:>14.4f} {t_np / t_c:>15.1f}")
        else:
            print(f"{p:>6} {t_np:>12.4f} {'n/a':>14} {'n/a':>15}")

    print("\nswapping_css end-to-end (p=100, k=10, single init)")
    t_np = bench_swapping(np_cholupdate)
    print(f"numpy backend:    {t_np:.3f} s")
    if c_cholupdate is not None:
        t_c = bench_swapping(c_cholupdate)
        print(f"choldate backend: {t_c:.3f} s")


if __name__ == "__main__":
    main()
