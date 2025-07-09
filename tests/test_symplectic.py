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
    assert symplectic.check(symplectic.givens(5, 0, -3.0))
    assert symplectic.check(symplectic.givens(2, 0, 100.0))


def test_householder_is_symplectic():
    def v(n, j):
        return np.concat([np.zeros(j - 1), np.arange(n - j + 1) + 1])

    assert symplectic.check(symplectic.householder(6, 2, v(6, 2), 0.4))
    assert symplectic.check(symplectic.householder(3, 1, v(3, 1), -1))
    assert symplectic.check(symplectic.householder(7, 6, v(7, 6), 100))
