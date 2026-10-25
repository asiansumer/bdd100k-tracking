import numpy as np


def kf_static_step(x, P, z, Q, R):
    # 预测：东西不动，位置不变，但没把握程度要加上 Q
    x_pred = x
    P_pred = P + Q
    # 卡尔曼增益：自己算出来的 α
    K = P_pred / (P_pred + R)
    # 修正位置：和 α-β 一样，往测量方向挪 K 那么多
    x_new = x_pred + (z - x_pred) * K
    # 用了这次测量，没把握程度变小
    P_new = (1 - K) * P_pred
    return x_new, P_new, K


def kf_step(s, P, z, F, H, Q, R):
    # ---- 预测 ----
    # 状态按 F 往前推一步
    s_pred = F @ s
    # 速度的没把握会漏进位置，再加上模型误差 Q
    P_pred = F @ P @ F.T + Q

    # ---- 更新 ----
    # 残差：测量比预测的位置多多少
    r = z - H @ s_pred
    # 残差本身有多没把握，相当于一维里的 P + R
    S = H @ P_pred @ H.T + R
    # 卡尔曼增益，相当于一维里的 P / (P + R)，inv 就是“除以”
    K = P_pred @ H.T @ np.linalg.inv(S)
    # 位置和速度各挪各的
    s_new = s_pred + K @ r
    # 用了测量，没把握程度变小，相当于一维里的 (1 - K) * P
    P_new = (np.eye(len(s)) - K @ H) @ P_pred
    return s_new, P_new, K


def box_matrices(dt=1):
    # F：8×8。先放对角线上的 1，意思是“每个数先保留自己”
    F = np.eye(8)
    # 前 4 个数（cx, cy, a, h）再各加上 dt 份自己的速度
    # 速度在后 4 格，所以第 i 个数的速度在第 i + 4 格
    for i in range(4):
        F[i, i + 4] = dt
    # H：检测器只给前 4 个数 cx, cy, a, h，不给速度
    H = np.eye(4, 8)
    return F, H


def box_noise(h):
    # 位置的典型误差：框高的 1/20
    std_pos = h / 20
    # 速度的典型误差：框高的 1/160
    std_vel = h / 160
    # Q：预测模型每帧多出的不靠谱，顺序 [cx, cy, a, h, vcx, vcy, va, vh]
    # 宽高比 a 几乎不变，给很小的固定值
    Q = np.diag([std_pos, std_pos, 1e-2, std_pos,
                 std_vel, std_vel, 1e-5, std_vel]) ** 2
    # R：检测器有多不准，顺序 [cx, cy, a, h]
    R = np.diag([std_pos, std_pos, 1e-1, std_pos]) ** 2
    return Q, R
