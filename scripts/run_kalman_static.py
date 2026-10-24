from sim1d import add_noise
from kalman import kf_static_step

# 一面不动的墙，真实距离 5 米，测 20 次
true_x = 5
n = 20
sigma = 0.5
# 传感器的没把握程度：sigma 的平方
R = sigma ** 2
Q = 0

# 墙不动，所以 20 个真实值都是 5
zs = add_noise([true_x] * n, sigma, seed=0)

# 一开始完全不知道墙在哪：随便猜 0，没把握程度给很大
x = 0
P = 100
Ks = []
for z in zs:
    x, P, K = kf_static_step(x, P, z, Q, R)
    Ks.append(K)

# 每个 K 保留两位小数再打印
print([round(k, 2) for k in Ks])
print("final estimate:", x)
