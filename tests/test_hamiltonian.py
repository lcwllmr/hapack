import pytest
import numpy as np
from hapack import symplectic, hessenberg, hamiltonian


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


@pytest.mark.parametrize("n", [1, 3, 5, 8])
def test_pvl_random(n: int):
    a = np.random.normal(size=(n, n))
    g = np.random.normal(size=(n, n))
    q = np.random.normal(size=(n, n))
    gasym = 0.5 * g - 0.5 * g.T
    qasym = 0.5 * q - 0.5 * q.T
    w = np.block([[a, gasym], [qasym, a.T]])

    u, pvl = hamiltonian.pvl(w)
    assert symplectic.check(u)
    assert np.allclose(np.eye(2 * n), u.T @ u)
    assert hamiltonian.check_skew(pvl)
    assert np.allclose(u.T @ w @ u, pvl)

    sw = pvl[n:, :n]
    assert np.allclose(0, sw)

    nw = pvl[:n, :n]
    assert hessenberg.check_upper(nw)

    se = pvl[n:, n:]
    assert hessenberg.check_lower(se)
