from sim1d import make_truth, add_noise


def test_make_truth():
    assert make_truth(0, 2, 0.5, 5) == [0, 1, 2, 3, 4]


def test_add_noise_zero_sigma():
    # sigma 为 0 时没有误差，测量值应该等于真实值
    assert add_noise([0, 1, 2], 0) == [0, 1, 2]


def test_add_noise_same_seed():
    # 同一个种子跑两次，结果应该一模一样
    assert add_noise([0, 1, 2], 0.5, seed=3) == add_noise([0, 1, 2], 0.5, seed=3)
