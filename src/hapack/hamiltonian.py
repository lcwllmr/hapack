import numpy as np
from hapack import symplectic


def check(a: np.ndarray) -> np.bool:
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


def check_skew(a: np.ndarray) -> np.bool:
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
