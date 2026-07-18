# CSS

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
