from hapack import symplectic

def test_structure_matrix_is_symplectic():
    j = symplectic.structure_matrix(5)
    assert symplectic.check(j)
