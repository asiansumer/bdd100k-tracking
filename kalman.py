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
    P_new = (np.eye(2) - K @ H) @ P_pred
    return s_new, P_new, K
