import numpy as np


def structure_matrix(n: int) -> np.ndarray:
    """
    Returns the canonical 2n-by-2n symplectic matrix.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    z = np.zeros((n, n))
    id = np.eye(n)
    return np.block([[z, id], [-id, z]])


def check(a: np.ndarray) -> bool:
    """
    Test whether a given matrix is symplectic.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] % 2 != 0:
        raise ValueError(
            f"a must be a 2n-by-2n square matrix but found shape {a.shape}"
        )
    n = a.shape[0] // 2
    j = structure_matrix(n)
    jtaj = j.T @ a @ j
    return np.allclose(jtaj, a)


def givens(n: int, j: int, theta: float) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be a positive integer")
    if not 0 <= j < n:
        raise ValueError("index j must be in the interval [0, n)")
    g = np.eye(2 * n)
    c, s = np.cos(theta), np.sin(theta)
    g[j, j] = c
    g[n + j, n + j] = c
    g[j, n + j] = s
    g[n + j, j] = -s
    return g


def householder(n: int, j: int, v: np.ndarray, beta: float) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be a positive integer")
    if v.ndim != 1 or v.size != n:
        raise ValueError(
            f"v must be 1-dimensional of size n={n} but found shape {v.shape}"
        )
    if not 0 <= j < n:
        raise ValueError("index j must be in the interval [0, n)")
    if j > 0 and not np.allclose(v[: j - 1], 0):
        raise ValueError("first j-1 elements of v are not all zero")
    z = np.zeros((n, n))
    h = np.eye(n) - beta * np.outer(v, v)
    return np.block([[h, z], [z, h]])


def elementary_projection(n: int, j: int, x: np.ndarray) -> np.ndarray:
    part2 = x[n:]
    v = np.zeros(n)
    v[j:] = part2[j:]
    v[j] -= np.linalg.norm(v)
    if np.isclose(0, np.dot(v, v)):
        beta = 0.0
    else:
        beta = 2 / np.dot(v, v)
    h1 = householder(n, j, v, beta)
    x = h1 @ x

    theta = np.arccos(x[j] / np.sqrt(x[j] ** 2 + x[n + j] ** 2))
    g = givens(n, j, theta)
    x = g @ x

    part1 = x[:n]
    w = np.zeros(n)
    w[j:] = part1[j:]
    w[j] -= np.linalg.norm(w)
    if np.isclose(0, np.dot(w, w)):
        gamma = 0.0
    else:
        gamma = 2 / np.dot(w, w)
    h2 = householder(n, j, w, gamma)

    return h2 @ g @ h1
