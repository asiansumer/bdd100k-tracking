def mean_abs_error(a, b):
    # 两个列表一一对应，平均每个差了多少（不管正负）
    total = 0.0
    for i in range(len(a)):
        # 第 i 个差了多少，加进总数
        total += abs(a[i] - b[i])
    return total / len(a)
