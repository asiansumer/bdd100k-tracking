from pytest import approx

from alphabeta import alpha_beta_step


def test_alpha_half():
    # 手算例子：预测 11，测量 12，挪一半到 11.5；beta 为 0 时速度不变
    x_new, v_new = alpha_beta_step(10, 2, 12, 0.5, 0, 0.5)
    assert x_new == approx(11.5)
    assert v_new == approx(2)


def test_beta_speeds_up():
    # 手算例子：速度估小了，残差为正，速度从 1 变成 1.2
    x_new, v_new = alpha_beta_step(10, 1, 11, 0.5, 0.2, 0.5)
    assert v_new == approx(1.2)
