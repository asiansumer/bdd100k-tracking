import numpy as np

def make_truth(x0, v, dt, n):
    # 存放每一时刻的真实位置
    xs = []
    for i in range(n):
        # 第 i 个点对应的时间
        t = i * dt
        xs.append(x0 + v * t)
    return xs


def add_noise(xs, sigma, seed=0):
    # 固定种子，每次运行得到同样的随机数
    rng = np.random.default_rng(seed)
    zs = []
    for x in xs:
        # 抽一个随机误差：平均 0，典型大小 sigma
        e = rng.normal(0, sigma)
        zs.append(x + e)
    return zs
