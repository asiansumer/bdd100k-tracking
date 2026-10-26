import pytest

from motion_models import NoMotion, AlphaBeta, Kalman
from tracker import Tracker, expand, greedy_match


def test_expand():
    # 宽 10、高 20 的框，四周各撑大 0.5 倍：左右各 5，上下各 10
    assert expand((0, 0, 10, 20), 0.5) == (-5, -10, 15, 30)


def test_greedy_match_picks_biggest_iou():
    m = NoMotion()
    tracks = {1: {"state": m.init((0, 0, 10, 10))}}
    boxes = [(5, 0, 15, 10), (1, 0, 11, 10)]
    matches, left_t, left_b = greedy_match([1], tracks, boxes, m, 0.3)
    # 第 1 个框（下标 1）重叠更多，被认领；下标 0 剩下
    assert matches == [(1, 1)]
    assert left_t == []
    assert left_b == [0]


@pytest.mark.parametrize("Model", [NoMotion, AlphaBeta, Kalman])
def test_one_walker_keeps_one_id(Model):
    # 一个人每帧往右走 3 像素，30 帧，编号应该一直是同一个
    tracker = Tracker(Model())
    ids = set()
    for k in range(30):
        box = (100 + 3 * k, 200, 130 + 3 * k, 280)
        for tid, _ in tracker.step([(box, 0.9)]):
            ids.add(tid)
    assert ids == {1}


def test_kalman_learns_speed():
    # 每帧右移 20 像素（5 帧/秒时很常见），几帧后预测应该跟得上
    m = Kalman()
    state = m.init((0, 0, 30, 80))
    for k in range(1, 6):
        state = m.predict(state)
        state = m.update(state, (20 * k, 0, 30 + 20 * k, 80))
    assert abs(state["s"][4] - 20) < 3
