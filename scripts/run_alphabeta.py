from sim1d import make_truth, add_noise
from alphabeta import alpha_beta_step
from metrics import mean_abs_error

import matplotlib.pyplot as plt


# 真实小车：起点 0，速度 2，每 0.5 秒一个点，共 40 个
v_true = 2
dt = 0.5
n = 40
sigma = 0.5
alpha = 0.5
beta = 0.1

xs = make_truth(0, v_true, dt, n)
zs = add_noise(xs, sigma, seed=0)

# 初始估计：位置用第一个测量，速度不知道，先当 0
x_est = zs[0]
v_est = 0
# 存每一步的位置估计，第一个就是初始值
est = [x_est]

# 从第 2 个测量开始，一个一个喂给滤波器
for z in zs[1:]:
    x_est, v_est = alpha_beta_step(x_est, v_est, z, alpha, beta, dt)
    est.append(x_est)

# 比一比：测量和 α-β 估计，平均各差真实位置多少米
print("measurement error:", mean_abs_error(zs, xs))
print("alpha-beta error:", mean_abs_error(est, xs))

# 每个点对应的时间，当作横轴
ts = [i * dt for i in range(n)]

fig, ax = plt.subplots()
ax.plot(ts, xs, label="truth")
ax.plot(ts, zs, ".", label="measurement")
# 滤波器的估计：也画成一条线
ax.plot(ts, est, label="alpha-beta")
ax.set_xlabel("time (s)")
ax.set_ylabel("position (m)")
ax.legend()
plt.show()
