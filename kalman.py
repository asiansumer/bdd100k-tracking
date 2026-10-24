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
