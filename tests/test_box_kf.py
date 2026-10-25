import numpy as np

from kalman import kf_step, box_matrices, box_noise


def test_box_kf_learns_speed():
    rng = np.random.default_rng(0)
    F, H = box_matrices()
    # 真实框：每帧 cx 加 3，其他不变
    truth = [np.array([100 + 3 * t, 200, 0.4, 100]) for t in range(30)]
    # 检测结果：真实框加上噪声
    zs = [b + rng.normal(0, [2, 2, 0.01, 2]) for b in truth]
    # 初始状态：框用第一次检测，速度全部猜 0
    s = np.concatenate([zs[0], np.zeros(4)])
    P = np.eye(8) * 10
    for z in zs[1:]:
        Q, R = box_noise(s[3])
        s, P, K = kf_step(s, P, z, F, H, Q, R)
    # 学出来的横向速度应该接近 3
    assert abs(s[4] - 3) < 0.5
