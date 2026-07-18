import numpy as np


def random_spd(p, seed):
    """A well-conditioned random symmetric positive-definite matrix."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((p, p))
    return M @ M.T + p * np.eye(p)


def upper_factor(A):
    """C-ordered upper-triangular R with A = R.T @ R."""
    return np.ascontiguousarray(np.linalg.cholesky(A).T)
