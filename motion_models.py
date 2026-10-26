"""三种运动模型，给跟踪器用。每种都提供同样的 4 个函数：

- init(box)：用第一个检测框建一个新的状态
- predict(state)：往前推一帧
- update(state, box)：用匹配上的检测框修正
- get_box(state)：取出当前的框 (x1, y1, x2, y2)

这样跟踪器的代码完全不用改，换一个模型就能公平比较。
"""
import numpy as np

from box_utils import xyxy_to_xyah, xyah_to_xyxy
from kalman import box_matrices, box_noise, kf_predict, kf_update

F, H = box_matrices()


class NoMotion:
    """不预测：框就停在上一次看到的位置。"""

    def init(self, box):
        return {"box": list(box)}

    def predict(self, state):
        return state

    def update(self, state, box):
        state["box"] = list(box)
        return state

    def get_box(self, state):
        return state["box"]


class AlphaBeta:
    """α-β：在 (cx, cy, a, h) 上每个数各做一个第 2 关的 α-β，dt = 1 帧。"""

    def __init__(self, alpha=0.9, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def init(self, box):
        return {"x": np.array(xyxy_to_xyah(box)), "v": np.zeros(4)}

    def predict(self, state):
        state["x"] = state["x"] + state["v"]
        return state

    def update(self, state, box):
        # 和第 2 关一样：残差 = 测量 − 预测，位置挪 alpha，速度挪 beta / dt（dt = 1）
        r = np.array(xyxy_to_xyah(box)) - state["x"]
        state["x"] = state["x"] + self.alpha * r
        state["v"] = state["v"] + self.beta * r
        return state

    def get_box(self, state):
        return xyah_to_xyxy(state["x"])


class Kalman:
    """卡尔曼：第 4 关的 8 维框卡尔曼。

    box_noise 是 ByteTrack 的设定，它是给每秒 30 帧的视频定的。
    BDD 的标注每秒只有 5 帧，一帧之间人能挪 6 倍远，所以要把两处“没把握”放大：
    - q_scale：Q 放大几倍（每一帧预测多出的不靠谱）
    - v0_std：新轨迹的速度一开始有多没把握，单位是“框高 / 帧”
    """

    def __init__(self, q_scale=4, v0_std=0.5):
        self.q_scale = q_scale
        self.v0_std = v0_std

    def init(self, box):
        z = np.array(xyxy_to_xyah(box))
        h = z[3]
        # 速度一开始当成 0
        s = np.concatenate([z, np.zeros(4)])
        # 位置：大约一个检测误差（框高的 1/10）；速度：大约 v0_std 个框高每帧
        std = [h / 10, h / 10, 1e-2, h / 10,
               self.v0_std * h, self.v0_std * h, 1e-5, self.v0_std * h]
        P = np.diag(std) ** 2
        return {"s": s, "P": P}

    def predict(self, state):
        Q, _ = box_noise(state["s"][3])
        # std 放大 q_scale 倍，方差就放大 q_scale² 倍
        Q = Q * self.q_scale ** 2
        state["s"], state["P"] = kf_predict(state["s"], state["P"], F, Q)
        return state

    def update(self, state, box):
        _, R = box_noise(state["s"][3])
        z = np.array(xyxy_to_xyah(box))
        state["s"], state["P"] = kf_update(state["s"], state["P"], z, H, R)
        return state

    def get_box(self, state):
        return xyah_to_xyxy(state["s"][:4])
