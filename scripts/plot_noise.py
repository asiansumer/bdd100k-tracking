from sim1d import make_truth, add_noise

import matplotlib.pyplot as plt

# 小车参数：起点 0 米，速度 2 米/秒，每 0.5 秒记一次，一共 40 个点
x0 = 0
v = 2
dt = 0.5
n = 40
# 测量误差的典型大小：0.5 米
sigma = 0.5

# 第 1 步：算出真实位置
xs = make_truth(x0, v, dt, n)
# 第 2 步：给真实位置加噪声，得到测量值
zs = add_noise(xs, sigma, seed=0)
ts = [i * dt for i in range(n)]

fig, ax = plt.subplots()
# 真实位置：画成一条线
ax.plot(ts, xs, label="truth")
# 测量值：只画点，不连线
ax.plot(ts, zs, ".", label="measurement")
ax.set_xlabel("time (s)")
ax.set_ylabel("position (m)")
ax.legend()
plt.show()
