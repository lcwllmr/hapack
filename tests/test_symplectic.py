import numpy as np
from hapack import symplectic


def test_identity_is_symplectic():
    id = np.eye(2 * 7)
    assert symplectic.check(id)


def test_structure_matrix_is_symplectic():
    j = symplectic.structure_matrix(5)
    assert symplectic.check(j)
