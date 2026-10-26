def xyxy_to_xyah(box):
    # 拆开四个数：左上角 (x1, y1)，右下角 (x2, y2)
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x2 + x1) / 2
    cy = (y2 + y1) / 2
    a = w / h
    return [cx, cy, a, h]


def xyah_to_xyxy(xyah):
    cx, cy, a, h = xyah
    # a = w / h，反过来求 w
    w = a * h
    # 中心往左半个宽、往上半个高，就是左上角
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)
    return [x1, y1, x2, y2]


def iou(a, b):
    """两个 (x1, y1, x2, y2) 框的交并比，0 到 1 之间。和 P1、P2 里的写法一样。"""
    # 交集框：起点取 max，终点取 min
    inter_w = max(min(a[2], b[2]) - max(a[0], b[0]), 0)
    inter_h = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    s_inter = inter_w * inter_h
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])
    s_union = s_a + s_b - s_inter
    if s_union == 0:
        return 0.0
    return s_inter / s_union
