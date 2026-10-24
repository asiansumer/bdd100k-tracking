def alpha_beta_step(x, v, z, alpha, beta, dt):
    # 1. 预测：按当前速度往前推一步
    x_pred = x + v * dt
    # 2. 残差：测量比预测多出来多少
    r = z - x_pred
    # 3. 修正位置：往测量方向挪 alpha 那么多
    x_new = x_pred + alpha * r
    # 4. 修正速度：速度差信 beta 那么多
    v_new = v + (beta / dt) * r
    return x_new, v_new
