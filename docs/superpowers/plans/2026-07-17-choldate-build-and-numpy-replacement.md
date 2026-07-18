# choldate Build Fix + Pure-NumPy Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pycss fork build cleanly with uv, vendor a build-fixed copy of the original Cython `choldate` package as a dev dependency, verify a pure-NumPy `cholupdate`/`choldowndate` replacement against it, and record reference benchmarks.

**Architecture:** The original `choldate` (Cython) gets vendored into `vendor/choldate/` with PEP 518 build fixes taken from upstream PR modusdatascience/choldate#8, wired in as a uv dev-only dependency so it never becomes a runtime requirement. `pycss/chol.py` is rewritten to be an exact drop-in for `choldate`'s API (in-place on upper-triangular `R`, mutates `x`, returns `None`, numerically stable `hypot`). Parity tests compare the two implementations and both against direct re-factorization; a benchmark script records relative performance.

**Tech Stack:** Python ≥3.10 (uv-managed), setuptools + Cython (vendored package only), NumPy, pytest.

## Global Constraints

- All environment management through `uv` — never conda (`environment.yml`/`requirements.txt` are upstream leftovers; do not use or update them).
- `choldate` must remain a **dev-only** dependency; `pycss`'s runtime Cholesky code stays pure NumPy.
- Working directory for every command: `/Users/jsperger/code/column-subset-selection-recipe/CSS-column-subset-selection` (this repo — the workspace root above it is not a git repo).
- Pre-existing test failures (if the baseline run in Task 1 shows any) are recorded, not fixed, in this plan.
- Do not commit `__pycache__`/`.pyc`/build artifacts (some are already tracked — leave them alone, but don't add more).
- Optimization is not a goal; benchmarks are for reference only.

---

### Task 1: uv project baseline

Make the fork a clean uv-managed project and establish the test baseline before touching choldate.

**Files:**
- Modify: `pyproject.toml`
- Create: `.python-version`, `uv.lock` (generated)

**Interfaces:**
- Produces: a working `.venv` where `uv run pytest` executes the existing suite; later tasks assume `uv sync` and `uv run` work from the repo root.

- [ ] **Step 1: Update `pyproject.toml`**

`pycss` is pure Python, so drop `Cython` from both the build requires and the dependency list (it was only ever needed for choldate), and add `requires-python`. Replace the whole file with:

```toml
[project]
name = "pycss"
version = "0.0.9"
description = "CSS Package"
authors = [{ name = "Anav Sood", email = "anavsood@gmail.com" }]
requires-python = ">=3.10"
dependencies = [
    "scipy",
    "numpy",
    "matplotlib",
    "tqdm",
    "pandas",
    "seaborn",
    "pytest",
    "jupyter",
    "cvxpy",
    "scs",
    "pingouin",
    "clarabel",
    "mosek",
]

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["pycss"]
```

- [ ] **Step 2: Pin a Python for uv**

```bash
echo "3.12" > .python-version
```

- [ ] **Step 3: Create the environment and install**

```bash
uv sync
```

Expected: uv creates `.venv`, resolves dependencies, writes `uv.lock`, and installs `pycss` itself (editable by default for the project). If `mosek` fails to resolve for the pinned Python, report back rather than working around it — it's an upstream dependency choice.

- [ ] **Step 4: Verify the package imports**

```bash
uv run python -c "from pycss.CSS import CSS; from pycss.chol import cholupdate; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Run the existing test suite to record the baseline**

```bash
uv run pytest tests/ -q
```

Expected: tests run. Record pass/fail counts. If there are failures, copy the failing test names into the commit message body as "pre-existing baseline failures" — do **not** fix them in this plan.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version uv.lock
git commit -m "build: make pycss a uv-managed project; drop unused Cython build dep"
```

---

### Task 2: Vendor choldate with a working build

Vendor the original Cython implementation into `vendor/choldate/` with the PEP 518 fixes from upstream PR #8 (rewritten more cleanly: static build requires in `pyproject.toml` instead of PR #8's `pip install` fallbacks inside `setup.py`), and install it as a uv dev dependency.

**Files:**
- Create: `vendor/choldate/pyproject.toml`
- Create: `vendor/choldate/setup.py`
- Create: `vendor/choldate/choldate/__init__.py`
- Create: `vendor/choldate/choldate/_choldate.pyx` (copied verbatim from upstream)
- Create: `vendor/choldate/LICENSE` (copied from upstream)
- Modify: `pyproject.toml` (add dev group + uv source)
- Delete: `choldate/` (stale empty directory left over from the removed submodule)

**Interfaces:**
- Produces: importable `choldate` module in the dev environment with `cholupdate(R, x)` and `choldowndate(R, x)` — both operate **in place** on a C-ordered upper-triangular float64 factor `R` (where `A = R.T @ R`), also mutate `x` in place, and return `None`. Tasks 3, 4, 5, and 7 import it exactly as `from choldate import cholupdate, choldowndate`.

- [ ] **Step 1: Fetch upstream source**

```bash
git clone --depth 1 https://github.com/modusdatascience/choldate.git /private/tmp/claude-501/-Users-jsperger-code-column-subset-selection-recipe/b586afd9-f053-441d-ac82-b70f702e35bb/scratchpad/choldate-upstream
```

Expected: clone succeeds. Inspect `choldate-upstream/choldate/_choldate.pyx` exists.

- [ ] **Step 2: Copy the source files we're vendoring**

Copy only the `.pyx` and license — we write our own packaging and `__init__.py` (upstream's may carry Python-2-era code we don't want):

```bash
mkdir -p vendor/choldate/choldate
cp /private/tmp/claude-501/-Users-jsperger-code-column-subset-selection-recipe/b586afd9-f053-441d-ac82-b70f702e35bb/scratchpad/choldate-upstream/choldate/_choldate.pyx vendor/choldate/choldate/_choldate.pyx
cp /private/tmp/claude-501/-Users-jsperger-code-column-subset-selection-recipe/b586afd9-f053-441d-ac82-b70f702e35bb/scratchpad/choldate-upstream/LICENSE* vendor/choldate/ 2>/dev/null || cp /private/tmp/claude-501/-Users-jsperger-code-column-subset-selection-recipe/b586afd9-f053-441d-ac82-b70f702e35bb/scratchpad/choldate-upstream/COPYING* vendor/choldate/LICENSE 2>/dev/null
ls vendor/choldate/
```

Expected: `_choldate.pyx` and a license file present. If no license file exists upstream, note that in the commit message and skip it (the repo is marked BSD-3-Clause on GitHub).

- [ ] **Step 3: Write `vendor/choldate/choldate/__init__.py`**

```python
from choldate._choldate import cholupdate, choldowndate

__all__ = ["cholupdate", "choldowndate"]
```

- [ ] **Step 4: Write `vendor/choldate/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64", "cython", "numpy"]
build-backend = "setuptools.build_meta"

[project]
name = "choldate"
version = "0.1.0+pycss.vendored"
description = "Rank-1 Cholesky update/downdate. Vendored from modusdatascience/choldate with the PEP 518 build fixes proposed in upstream PR #8."
requires-python = ">=3.10"
dependencies = ["numpy"]

[tool.setuptools]
packages = ["choldate"]
```

- [ ] **Step 5: Write `vendor/choldate/setup.py`**

```python
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "choldate._choldate",
                ["choldate/_choldate.pyx"],
                include_dirs=[numpy.get_include()],
            )
        ],
        language_level=3,
    ),
)
```

- [ ] **Step 6: Remove the stale submodule leftover**

```bash
rmdir choldate
```

Expected: succeeds (the directory is empty). If it is not empty, stop and look — that would mean untracked local files worth surfacing before deleting anything.

- [ ] **Step 7: Register choldate as a dev dependency in the root `pyproject.toml`**

Append to `pyproject.toml`:

```toml
[dependency-groups]
dev = ["choldate"]

[tool.uv.sources]
choldate = { path = "vendor/choldate" }
```

- [ ] **Step 8: Build and install**

```bash
uv sync
```

Expected: uv builds the `choldate` wheel from `vendor/choldate` (compiling the Cython extension) and installs it. This is the step the previous attempt never got working — if compilation fails, apply these contingencies in order, re-running `uv sync` after each:

1. Error mentions `abs` from `libc.math` (Cython 3 removed the ambiguous `abs`): edit `vendor/choldate/choldate/_choldate.pyx`, changing `from libc.math cimport abs as cabs` to `from libc.math cimport fabs as cabs`. This is semantically identical for the float arguments used.
2. Other Cython-3 syntax errors in the `.pyx`: pin the old toolchain instead of patching — in `vendor/choldate/pyproject.toml` change `"cython"` to `"cython<3"` in `[build-system].requires`.
3. Error mentions missing C compiler: run `xcode-select --install` and report to the user that Command Line Tools are needed.

Record which (if any) contingency was needed in the commit message.

- [ ] **Step 9: Smoke-test the import**

```bash
uv run python -c "from choldate import cholupdate, choldowndate; print('choldate ok')"
```

Expected: `choldate ok`

- [ ] **Step 10: Commit**

```bash
git add vendor/ pyproject.toml uv.lock
git commit -m "build: vendor choldate with PEP 518 build fixes as uv dev dependency

Vendors modusdatascience/choldate (BSD-3-Clause) with the build
modernization from upstream PR #8 so the original Cython implementation
builds under uv. Dev-only dependency: pycss runtime does not require it.
Replaces the git submodule approach from e8c0faa that never built."
```

---

### Task 3: Correctness tests for the vendored choldate

Prove the vendored build is actually correct (not just importable) by checking it against direct re-factorization. These tests also define the ground-truth harness reused in Task 4.

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_choldate_vendored.py`

**Interfaces:**
- Consumes: `from choldate import cholupdate, choldowndate` (Task 2).
- Produces: helper functions `random_spd(p, seed)` and `upper_factor(A)` with these exact signatures, which Task 4's tests import via `from tests.test_choldate_vendored import random_spd, upper_factor`.

- [ ] **Step 1: Make `tests/` an importable package**

Create an empty `tests/__init__.py` so test modules can import helpers from each other (pytest then puts the repo root — the package parent — on `sys.path`):

```bash
touch tests/__init__.py
```

- [ ] **Step 2: Write the test file**

```python
import numpy as np
import pytest

choldate = pytest.importorskip("choldate")


def random_spd(p, seed):
    """A well-conditioned random symmetric positive-definite matrix."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((p, p))
    return M @ M.T + p * np.eye(p)


def upper_factor(A):
    """C-ordered upper-triangular R with A = R.T @ R."""
    return np.ascontiguousarray(np.linalg.cholesky(A).T)


@pytest.mark.parametrize("p", [1, 2, 5, 20, 100])
def test_cholupdate_matches_refactorization(p):
    rng = np.random.default_rng(p)
    A = random_spd(p, seed=p)
    x = rng.standard_normal(p)

    R = upper_factor(A)
    choldate.cholupdate(R, x.copy())

    expected = upper_factor(A + np.outer(x, x))
    np.testing.assert_allclose(R, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("p", [1, 2, 5, 20, 100])
def test_choldowndate_matches_refactorization(p):
    # Downdate A + x x^T by x; must recover the factor of A (A stays PD).
    rng = np.random.default_rng(p)
    A = random_spd(p, seed=p)
    x = rng.standard_normal(p)

    R = upper_factor(A + np.outer(x, x))
    choldate.choldowndate(R, x.copy())

    expected = upper_factor(A)
    np.testing.assert_allclose(R, expected, rtol=1e-8, atol=1e-10)


def test_cholupdate_is_in_place_and_returns_none():
    A = random_spd(5, seed=0)
    x = np.arange(1.0, 6.0)
    R = upper_factor(A)
    x_before = x.copy()

    result = choldate.cholupdate(R, x)

    assert result is None
    assert not np.array_equal(x, x_before)  # choldate mutates x too
```

- [ ] **Step 3: Run the new tests**

```bash
uv run pytest tests/test_choldate_vendored.py -v
```

Expected: all PASS. A failure here means the vendored build is miscompiled or the upstream algorithm assumption is wrong — stop and investigate before proceeding (do not loosen tolerances to get to green).

- [ ] **Step 4: Confirm adding `tests/__init__.py` didn't break existing collection**

```bash
uv run pytest tests/ -q --collect-only
```

Expected: all existing test modules still collect (same module count as the Task 1 baseline).

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_choldate_vendored.py
git commit -m "test: verify vendored choldate against direct re-factorization"
```

---

### Task 4: Rewrite `pycss/chol.py` cholupdate as an exact choldate drop-in

The current `chol.py` differs from choldate observably: it copies `x` instead of mutating it, returns `R`, and uses the numerically naive `sqrt(a**2 + b**2)` instead of a stable `hypot`. Rewrite it to match choldate's contract exactly, test-first.

**Files:**
- Create: `tests/test_chol.py`
- Modify: `pycss/chol.py`

**Interfaces:**
- Consumes: `random_spd`, `upper_factor` from `tests/test_choldate_vendored.py` (Task 3); `from choldate import cholupdate` for parity tests.
- Produces: `pycss.chol.cholupdate(R, x)` — in place on upper-triangular C-ordered float64 `R` (`A = R.T @ R` becomes `A + x x^T`), mutates `x`, returns `None`. The existing call site `update_cholesky_after_removing_first` (`pycss/subset_selection.py:208`) already ignores the return value and passes a copied `x`, so it needs no change.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chol.py`:

```python
import numpy as np
import pytest

from pycss.chol import cholupdate
from tests.test_choldate_vendored import random_spd, upper_factor


@pytest.mark.parametrize("p", [1, 2, 5, 20, 100])
def test_cholupdate_matches_refactorization(p):
    rng = np.random.default_rng(p)
    A = random_spd(p, seed=p)
    x = rng.standard_normal(p)

    R = upper_factor(A)
    cholupdate(R, x.copy())

    expected = upper_factor(A + np.outer(x, x))
    np.testing.assert_allclose(R, expected, rtol=1e-8, atol=1e-10)


def test_cholupdate_is_in_place_and_returns_none():
    A = random_spd(5, seed=0)
    x = np.arange(1.0, 6.0)
    R = upper_factor(A)
    x_before = x.copy()

    result = cholupdate(R, x)

    assert result is None
    assert not np.array_equal(x, x_before)


@pytest.mark.parametrize("p", [1, 2, 5, 20, 100])
def test_cholupdate_parity_with_choldate(p):
    choldate = pytest.importorskip("choldate")
    rng = np.random.default_rng(p + 1000)
    A = random_spd(p, seed=p + 1000)
    x = rng.standard_normal(p)

    R_np, x_np = upper_factor(A), x.copy()
    R_c, x_c = upper_factor(A), x.copy()

    cholupdate(R_np, x_np)
    choldate.cholupdate(R_c, x_c)

    np.testing.assert_allclose(R_np, R_c, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(x_np, x_c, rtol=1e-12, atol=1e-14)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_chol.py -v
```

Expected: `test_cholupdate_is_in_place_and_returns_none` FAILS (current implementation returns `R` and does not mutate `x`); the parity test FAILS on `x` mutation. The refactorization test may pass — that's fine; the failing tests are what drive the rewrite.

- [ ] **Step 3: Rewrite `pycss/chol.py`**

Replace the file's `cholupdate` with:

```python
import numpy as np


def cholupdate(R, x):
    """Rank-1 update of an upper-triangular Cholesky factor, in place.

    Drop-in replacement for choldate.cholupdate: given upper-triangular
    `R` with `A = R.T @ R`, overwrites `R` with the factor of
    `A + np.outer(x, x)`. Both `R` and `x` are modified in place and
    nothing is returned, matching the choldate API exactly.
    """
    p = len(x)
    for k in range(p):
        r = np.hypot(R[k, k], x[k])
        c = r / R[k, k]
        s = x[k] / R[k, k]
        R[k, k] = r
        R[k, k + 1:] = (R[k, k + 1:] + s * x[k + 1:]) / c
        x[k + 1:] = c * x[k + 1:] - s * R[k, k + 1:]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_chol.py -v
```

Expected: all PASS (parity tests skip only if choldate failed to install, which Task 2 rules out).

- [ ] **Step 5: Run the full suite to catch call-site regressions**

```bash
uv run pytest tests/ -q
```

Expected: same results as the Task 1 baseline (no new failures). `update_cholesky_after_removing_first` is exercised via the swapping-CSS tests.

- [ ] **Step 6: Commit**

```bash
git add pycss/chol.py tests/test_chol.py
git commit -m "fix: make pure-NumPy cholupdate an exact choldate drop-in

Match choldate semantics (in-place on R and x, returns None) and use
np.hypot for numerical stability instead of naive sqrt of squares."
```

---

### Task 5: Add pure-NumPy choldowndate

pycss only calls `cholupdate` today, but the R port and full choldate replacement need the downdate too; it completes API parity with the vendored package.

**Files:**
- Modify: `pycss/chol.py`
- Modify: `tests/test_chol.py`

**Interfaces:**
- Consumes: `random_spd`, `upper_factor` helpers; `from choldate import choldowndate` for parity.
- Produces: `pycss.chol.choldowndate(R, x)` — same in-place contract as `cholupdate`, subtracting `np.outer(x, x)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chol.py` (and extend the top import to `from pycss.chol import cholupdate, choldowndate`):

```python
@pytest.mark.parametrize("p", [1, 2, 5, 20, 100])
def test_choldowndate_matches_refactorization(p):
    rng = np.random.default_rng(p)
    A = random_spd(p, seed=p)
    x = rng.standard_normal(p)

    R = upper_factor(A + np.outer(x, x))
    choldowndate(R, x.copy())

    expected = upper_factor(A)
    np.testing.assert_allclose(R, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("p", [1, 2, 5, 20, 100])
def test_choldowndate_parity_with_choldate(p):
    choldate = pytest.importorskip("choldate")
    rng = np.random.default_rng(p + 2000)
    A = random_spd(p, seed=p + 2000)
    x = rng.standard_normal(p)
    R0 = upper_factor(A + np.outer(x, x))

    R_np, x_np = R0.copy(), x.copy()
    R_c, x_c = R0.copy(), x.copy()

    choldowndate(R_np, x_np)
    choldate.choldowndate(R_c, x_c)

    np.testing.assert_allclose(R_np, R_c, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(x_np, x_c, rtol=1e-12, atol=1e-14)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_chol.py -v
```

Expected: the two new tests FAIL with `ImportError: cannot import name 'choldowndate' from 'pycss.chol'` (collection error is acceptable — it demonstrates the function doesn't exist).

- [ ] **Step 3: Implement `choldowndate` in `pycss/chol.py`**

Append:

```python
def choldowndate(R, x):
    """Rank-1 downdate of an upper-triangular Cholesky factor, in place.

    Drop-in replacement for choldate.choldowndate: given upper-triangular
    `R` with `A = R.T @ R`, overwrites `R` with the factor of
    `A - np.outer(x, x)` (which must remain positive definite). Both `R`
    and `x` are modified in place and nothing is returned.
    """
    p = len(x)
    for k in range(p):
        r = np.sqrt((R[k, k] - x[k]) * (R[k, k] + x[k]))
        c = r / R[k, k]
        s = x[k] / R[k, k]
        R[k, k] = r
        R[k, k + 1:] = (R[k, k + 1:] - s * x[k + 1:]) / c
        x[k + 1:] = c * x[k + 1:] - s * R[k, k + 1:]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_chol.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pycss/chol.py tests/test_chol.py
git commit -m "feat: add pure-NumPy choldowndate for full choldate API parity"
```

---

### Task 6: Reference benchmarks

Micro-benchmark `cholupdate` (NumPy vs Cython) and one end-to-end `swapping_css` run with each backend. Reference numbers only — no optimization work.

**Files:**
- Create: `benchmarks/bench_chol.py`
- Create: `benchmarks/RESULTS.md`

**Interfaces:**
- Consumes: `pycss.chol.cholupdate`, `choldate.cholupdate`, `pycss.subset_selection.swapping_css(Sigma, k, ...)`.
- Produces: standalone script; nothing downstream depends on it.

- [ ] **Step 1: Write `benchmarks/bench_chol.py`**

```python
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
```

- [ ] **Step 2: Run it**

```bash
uv run python benchmarks/bench_chol.py
```

Expected: the table prints with both columns populated (choldate importable) and both `swapping_css` runs return the same-length subset without error. If `swapping_css(Sigma, k, S_init=...)` raises a signature error, check `pycss/subset_selection.py:692` for the actual keyword names and adjust the call.

- [ ] **Step 3: Record results**

Create `benchmarks/RESULTS.md` containing the date (2026-07-17), the machine description (`uname -m`, macOS, Python version from `uv run python -V`), and the verbatim script output in a fenced code block.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/
git commit -m "bench: reference comparison of NumPy cholupdate vs Cython choldate"
```

---

### Task 7: Documentation sync

Update the fork's README and the workspace CLAUDE.md so the choldate story is accurate.

**Files:**
- Modify: `README.md` (fork root)
- Modify: `../CLAUDE.md` (workspace — not part of this git repo; edit but don't commit here)

**Interfaces:**
- Consumes: outcomes of Tasks 1–6 (which contingencies fired, benchmark summary).

- [ ] **Step 1: Replace the README installation section**

Replace both installation sections of `README.md` (the uv section's choldate/rpy2 warning and the entire conda section) with:

````markdown
## Installation

This fork uses [uv](https://docs.astral.sh/uv/). From the repo root:

```
uv sync
uv run pytest tests/
```

`pycss` itself is pure Python. The original Cython
[choldate](https://github.com/modusdatascience/choldate) dependency is
replaced by a pure-NumPy implementation in `pycss/chol.py`; the original
package is vendored under `vendor/choldate/` (with the build fixes from
[choldate PR #8](https://github.com/modusdatascience/choldate/pull/8)) as a
dev-only dependency so the replacement is tested against it
(`tests/test_chol.py`). Reference benchmarks live in `benchmarks/`.
````

- [ ] **Step 2: Update the workspace `CLAUDE.md`**

In `/Users/jsperger/code/column-subset-selection-recipe/CLAUDE.md`, rewrite "The choldate situation" section to reflect the new state: vendored build-fixed choldate at `vendor/choldate/` (dev-only), verified pure-NumPy `cholupdate`/`choldowndate` in `pycss/chol.py` with parity tests in `tests/test_chol.py`, benchmarks in `benchmarks/`. Update the Python commands section from `uv pip install -e .` to `uv sync`. Remove the "treat with skepticism / unverified" warnings — they will no longer be true.

- [ ] **Step 3: Commit the fork's README**

```bash
git add README.md
git commit -m "docs: document uv workflow and choldate replacement status"
```

---

## Verification (whole plan)

- `uv sync` from a clean checkout (delete `.venv`) succeeds and compiles the vendored choldate.
- `uv run pytest tests/ -q` — all new tests pass; no regressions vs the Task 1 baseline.
- `uv run python benchmarks/bench_chol.py` runs to completion with both backends.
