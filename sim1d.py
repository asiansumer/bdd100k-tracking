def make_truth(x0, v, dt, n):
    # 存放每一时刻的真实位置
    xs = []
    for i in range(n):
        # 第 i 个点对应的时间
        t = i * dt
        xs.append(x0 + v * t)
    return xs
