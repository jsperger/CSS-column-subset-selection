import numpy as np
import pytest

from tests.helpers import random_spd, upper_factor

choldate = pytest.importorskip("choldate")


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
