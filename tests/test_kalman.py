from pytest import approx

import numpy as np

from kalman import kf_static_step, kf_step


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


F = np.array([[1, 0.5], [0, 1]])
H = np.array([[1, 0]])
R = np.array([[1.0]])
Q0 = np.zeros((2, 2))


def test_kf_no_uncertainty_just_predicts():
    # P 为 0：对预测完全有把握，K 为 0，结果就是 F @ s
    s_new, P_new, K = kf_step(np.array([10, 2]), Q0, np.array([12]), F, H, Q0, R)
    assert s_new == approx([11, 2])


def test_kf_unsure_trusts_measurement():
    # P 非常大：完全没把握，位置几乎直接跳到测量 10
    s_new, P_new, K = kf_step(np.array([0, 0]), np.eye(2) * 1e6, np.array([10]), F, H, Q0, R)
    assert s_new[0] == approx(10, abs=0.01)


def test_kf_positive_residual_speeds_up():
    # 测量比预测多，速度应该往上调
    s_new, P_new, K = kf_step(np.array([10, 1]), np.eye(2), np.array([12]), F, H, Q0, R)
    assert s_new[1] > 1
