from pytest import approx

from metrics import mean_abs_error


def test_mean_abs_error():
    # 三个差分别是 +1、-1、0，绝对值的平均是 (1 + 1 + 0) / 3
    assert mean_abs_error([1, 2, 3], [0, 3, 3]) == approx(2 / 3)
