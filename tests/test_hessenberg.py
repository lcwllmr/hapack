import numpy as np
from hapack import hessenberg


def test_check_upper():
    a = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]])
    assert hessenberg.check_upper(a)

    a[2, 0] = 1
    assert not hessenberg.check_upper(a)


def test_check_lower():
    a = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    assert hessenberg.check_lower(a)

    a[0, 2] = 1
    assert not hessenberg.check_lower(a)
