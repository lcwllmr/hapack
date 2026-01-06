import numpy as np


def check_upper(a: np.ndarray) -> bool:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be a square matrix but found shape {a.shape}")
    n = a.shape[0]
    lower_indices = np.tril_indices(n, -2)
    return np.allclose(0, a[lower_indices])


def check_lower(a: np.ndarray) -> bool:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a must be a square matrix but found shape {a.shape}")
    n = a.shape[0]
    upper_indices = np.triu_indices(n, 2)
    return np.allclose(0, a[upper_indices])
