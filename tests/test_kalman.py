from pytest import approx

from kalman import kf_static_step


def test_first_step():
    # P=9，R=1，Q=0：K=0.9，从 0 往测量 10 挪 0.9 到 9，P 变成 0.9
    x_new, P_new, K = kf_static_step(0, 9, 10, 0, 1)
    assert K == approx(0.9)
    assert x_new == approx(9)
    assert P_new == approx(0.9)


def test_q_adds_uncertainty():
    # Q=1 时，预测后 P 从 9 变 10，K = 10 / 11
    x_new, P_new, K = kf_static_step(0, 9, 10, 1, 1)
    assert K == approx(10 / 11)
