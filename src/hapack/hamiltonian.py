import numpy as np
from hapack import symplectic


def check(a: np.ndarray) -> bool:
    """
    Test whether a given matrix is Hamiltonian.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] % 2 != 0:
        raise ValueError(
            f"a must be a 2n-by-2n square matrix but found shape {a.shape}"
        )
    n = a.shape[0] // 2
    aj = a @ symplectic.structure_matrix(n)
    return np.allclose(aj, aj.T)


def check_skew(a: np.ndarray) -> bool:
    """
    Test whether a given matrix is skew-Hamiltonian.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] % 2 != 0:
        raise ValueError(
            f"a must be a 2n-by-2n square matrix but found shape {a.shape}"
        )
    n = a.shape[0] // 2
    aj = a @ symplectic.structure_matrix(n)
    return np.allclose(aj, -aj.T)


def pvl(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not check_skew(w):
        raise ValueError("w must be a skew-Hamiltonian matrix")
    n = w.shape[0] // 2

    u = np.eye(2 * n)
    pvl = np.copy(w)
    for j in range(n - 1):
        ej = np.zeros(2 * n)
        ej[j] = 1.0
        x = pvl @ ej

        epj = symplectic.elementary_projection(n, j + 1, x).T
        pvl = epj.T @ pvl @ epj
        u = u @ epj

    return u, pvl
