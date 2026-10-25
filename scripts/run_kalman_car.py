import numpy as np
import matplotlib.pyplot as plt

from sim1d import make_truth, add_noise
from alphabeta import alpha_beta_step
from kalman import kf_step
from metrics import mean_abs_error

# 和 α-β 实验 C 完全一样的小车和测量
v_true = 2
dt = 0.5
n = 40
sigma = 0.5
xs = make_truth(0, v_true, dt, n)
zs = add_noise(xs, sigma, seed=0)

# ---- α-β：实验 C 的参数，初始速度错猜成 0 ----
x_est = zs[0]
v_est = 0
ab = [x_est]
for z in zs[1:]:
    x_est, v_est = alpha_beta_step(x_est, v_est, z, 0.1, 0.01, dt)
    ab.append(x_est)

# ---- 卡尔曼：初始速度同样猜 0 ----
F = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])
# 传感器的没把握：sigma 的平方
R = np.array([[sigma ** 2]])
# 模型误差：车基本匀速，给一个很小的值
Q = np.eye(2) * 0.001
# 初始状态：位置用第一个测量，速度猜 0
s = np.array([zs[0], 0.0])
# 初始没把握：位置和一次测量一样准，速度完全不知道，给很大
P = np.array([[sigma ** 2, 0], [0, 100.0]])
kf = [s[0]]
# 记下每一步“位置挪几成”，看它怎么变
Kx = []
for z in zs[1:]:
    s, P, K = kf_step(s, P, z, F, H, Q, R)
    kf.append(s[0])
    Kx.append(K[0, 0])

print("measurement error:", mean_abs_error(zs, xs))
print("alpha-beta (C) error:", mean_abs_error(ab, xs))
print("kalman error:", mean_abs_error(kf, xs))
print("K_x first 5:", [round(k, 2) for k in Kx[:5]])
print("K_x last:", round(Kx[-1], 2))
print("final speed estimate:", s[1])

ts = [i * dt for i in range(n)]
fig, ax = plt.subplots()
ax.plot(ts, xs, label="truth")
ax.plot(ts, zs, ".", label="measurement")
ax.plot(ts, ab, label="alpha-beta (C)")
ax.plot(ts, kf, label="kalman")
ax.set_xlabel("time (s)")
ax.set_ylabel("position (m)")
ax.legend()
plt.show()
