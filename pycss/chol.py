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
