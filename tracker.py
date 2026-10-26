"""ByteTrack 式的跟踪器（P2 的 v4 改写），运动模型可以换。

每一帧做 5 件事：
1. 所有老轨迹先用运动模型往前推一帧（预测）
2. 第一轮匹配：高分检测框 ↔ 所有轨迹
3. 第二轮匹配：低分检测框 ↔ 第一轮没匹配上的轨迹（被挡住时分数会变低，用它们续命）
4. 匹配上的轨迹用检测框更新；没匹配上的记一次“丢失”，丢太久就删掉
5. 没被认领的高分框开新轨迹

和 P2 不一样的地方：BDD 每秒只有 5 帧，车也在动，同一个人前后两帧的框经常几乎不重叠。
所以算 IoU 之前先把两个框都撑大一圈（“缓冲 IoU”，C-BIoU 论文的做法），让离得不远的框也能配上。
"""
from box_utils import iou


def expand(box, b):
    """把框往四周撑大：左右各加 b 倍宽，上下各加 b 倍高。"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return (x1 - b * w, y1 - b * h, x2 + b * w, y2 + b * h)


def greedy_match(track_ids, tracks, boxes, motion, iou_thr, buffer=0.0):
    """把所有 (轨迹, 检测框) 配对按 IoU 从大到小排，依次认领，一个只能配一次。

    返回 (配对列表 [(轨迹编号, 框下标)], 没配上的轨迹编号, 没配上的框下标)。
    """
    pairs = []
    for tid in track_ids:
        pred = motion.get_box(tracks[tid]["state"])
        for j, box in enumerate(boxes):
            v = iou(expand(pred, buffer), expand(box, buffer))
            if v >= iou_thr:
                pairs.append((v, tid, j))
    pairs.sort(reverse=True)

    used_t, used_b, matches = set(), set(), []
    for v, tid, j in pairs:
        if tid in used_t or j in used_b:
            continue
        used_t.add(tid)
        used_b.add(j)
        matches.append((tid, j))
    left_t = [t for t in track_ids if t not in used_t]
    left_b = [j for j in range(len(boxes)) if j not in used_b]
    return matches, left_t, left_b


class Tracker:
    def __init__(self, motion, high_thr=0.4, new_thr=0.5, iou_thr=0.2, low_iou_thr=0.5,
                 max_age=5, buffer=0.6):
        self.motion = motion
        self.high_thr = high_thr
        self.new_thr = new_thr
        self.iou_thr = iou_thr
        self.low_iou_thr = low_iou_thr
        self.max_age = max_age
        self.buffer = buffer
        # 档案：{编号: {"state": 运动模型的状态, "missed": 连续没看到几帧, "hits": 一共匹配上几次}}
        self.tracks = {}
        self.next_id = 1

    def step(self, dets):
        """处理一帧。dets 是 [(框, 分数), ...]。返回这一帧要输出的 [(编号, 框), ...]。"""
        m = self.motion
        # 1. 预测
        for t in self.tracks.values():
            t["state"] = m.predict(t["state"])

        high = [b for b, s in dets if s >= self.high_thr]
        low = [b for b, s in dets if s < self.high_thr]
        high_scores = [s for b, s in dets if s >= self.high_thr]

        # 2. 第一轮：高分框
        ids = list(self.tracks)
        matches, left_t, left_high = greedy_match(ids, self.tracks, high, m, self.iou_thr, self.buffer)
        for tid, j in matches:
            self._hit(tid, high[j])

        # 3. 第二轮：低分框只给已经确认过的老轨迹续命，要求重叠更多
        confirmed = [t for t in left_t if self.tracks[t]["hits"] >= 2]
        matches2, left_t2, _ = greedy_match(confirmed, self.tracks, low, m, self.low_iou_thr)
        for tid, j in matches2:
            self._hit(tid, low[j])
        lost = set(left_t2) | (set(left_t) - set(confirmed))

        # 4. 没匹配上的：记一次丢失；刚出现一次就丢的（多半是误检）直接删
        for tid in lost:
            t = self.tracks[tid]
            t["missed"] += 1
            if t["hits"] < 2 or t["missed"] > self.max_age:
                del self.tracks[tid]

        # 5. 没被认领、分数够高的框开新轨迹
        for j in left_high:
            if high_scores[j] >= self.new_thr:
                self.tracks[self.next_id] = {"state": m.init(high[j]), "missed": 0, "hits": 1}
                self.next_id += 1

        # 输出：这一帧匹配上、并且至少匹配过 2 次的轨迹（只出现 1 次的先不报）
        out = []
        for tid, t in self.tracks.items():
            if t["missed"] == 0 and t["hits"] >= 2:
                out.append((tid, m.get_box(t["state"])))
        return out

    def _hit(self, tid, box):
        t = self.tracks[tid]
        t["state"] = self.motion.update(t["state"], box)
        t["missed"] = 0
        t["hits"] += 1
