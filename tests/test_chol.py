import numpy as np
import pytest

from pycss.chol import cholupdate, choldowndate
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
