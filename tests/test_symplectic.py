import numpy as np
from hapack import symplectic


def test_identity_is_symplectic():
    id = np.eye(2 * 7)
    assert symplectic.check(id)


def test_structure_matrix_is_symplectic():
    j = symplectic.structure_matrix(5)
    assert symplectic.check(j)


def test_givens_is_symmetric():
    assert symplectic.check(symplectic.givens(4, 0, 0.32))
    assert symplectic.check(symplectic.givens(5, 0, -3.0))
    assert symplectic.check(symplectic.givens(2, 0, 100.0))
