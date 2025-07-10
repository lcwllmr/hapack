import pytest
import numpy as np
from hapack import symplectic, hamiltonian


def test_check_identity():
    assert not hamiltonian.check(np.eye(2 * 4))
    assert hamiltonian.check_skew(np.eye(2 * 5))


def test_check_structure():
    j = symplectic.structure_matrix(6)
    assert hamiltonian.check(j)
    assert not hamiltonian.check_skew(j)


@pytest.mark.parametrize("n", [1, 3, 5, 8])
def test_check_random(n: int):
    a = np.random.normal(size=(n, n))
    g = np.random.normal(size=(n, n))
    q = np.random.normal(size=(n, n))

    gsym = 0.5 * g + 0.5 * g.T
    qsym = 0.5 * q + 0.5 * q.T
    h = np.block([[a, gsym], [qsym, -a.T]])
    assert hamiltonian.check(h)

    gasym = 0.5 * g - 0.5 * g.T
    qasym = 0.5 * q - 0.5 * q.T
    sh = np.block([[a, gasym], [qasym, a.T]])
    assert hamiltonian.check_skew(sh)
