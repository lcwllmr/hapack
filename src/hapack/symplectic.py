import numpy as np

def structure_matrix(n: int) -> np.ndarray:
    """
    Returns the canonical 2n-by-2n symplectic matrix.
    """
    if n < 1:
        raise ValueError('n must be a positive integer')
    z = np.zeros((n,n))
    id = np.eye(n)
    return np.block([[z, id], [-id, z]])

def check(a: np.ndarray) -> np.bool:
    """
    Test whether a given matrix is symplectic.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] % 2 != 0:
        raise ValueError(f'a must be a 2n-by-2n square matrix but found shape {a.shape}')
    n = a.shape[0] // 2
    j = structure_matrix(n)
    jtaj = j.T @ a @ j
    return np.allclose(jtaj, a)
    
    
    
