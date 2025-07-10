import pytest
import numpy as np
from hapack import symplectic


def test_identity_is_symplectic():
    assert symplectic.check(np.eye(2 * 7))
    assert symplectic.check(np.eye(2 * 3))


def test_structure_matrix_is_symplectic():
    assert symplectic.check(symplectic.structure_matrix(4))
    assert symplectic.check(symplectic.structure_matrix(5))


def test_givens_is_symplectic():
    assert symplectic.check(symplectic.givens(4, 0, 0.32))
    assert symplectic.check(symplectic.givens(5, 3, -3.0))
    assert symplectic.check(symplectic.givens(2, 1, 100.0))


def test_givens_is_orthogonal():
    g1 = symplectic.givens(4, 2, 1.5)
    assert np.allclose(np.eye(8), g1.T @ g1)
    g2 = symplectic.givens(7, 0, -1.5)
    assert np.allclose(np.eye(14), g2.T @ g2)


def test_householder_is_symplectic():
    def v(n, j):
        return np.concat([np.zeros(j - 1), np.arange(n - j + 1) + 1])

    assert symplectic.check(symplectic.householder(6, 2, v(6, 2), 0.4))
    assert symplectic.check(symplectic.householder(3, 1, v(3, 1), -1))
    assert symplectic.check(symplectic.householder(7, 6, v(7, 6), 100))


def test_householder_is_orthogonal():
    n = 5
    j = 3
    v = np.concat([np.zeros(j - 1), np.arange(n - j + 1) + 1])
    v = v / np.linalg.norm(v)
    beta = 2
    h = symplectic.householder(n, j, v, beta)
    assert np.allclose(np.eye(2 * n), h.T @ h)


@pytest.mark.parametrize("j", range(7))
def test_elementary_projection(j: int):
    n = 7
    x = np.random.normal(size=(2 * n,))
    e = symplectic.elementary_projection(n, j, x)
    y = e @ x

    assert np.allclose(x[:j], y[:j])
    assert np.allclose(0, y[j + 1 : n])
    assert np.allclose(x[n : n + j], y[n : n + j])
    assert np.allclose(0, y[n + j :])
