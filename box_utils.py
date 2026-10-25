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
