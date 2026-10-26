import pandas as pd
import pytest

from bdd_io import load_ped_labels
from box_utils import iou
from scripts.prepare_bdd import video_index


def test_iou():
    a = (0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, (20, 20, 30, 30)) == 0.0
    # 重叠一半：交集 50，并集 150
    assert iou(a, (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_video_index():
    # 30 帧/秒的视频、5 帧/秒的标注：每个标注帧隔 6 个视频帧
    assert video_index(2, 30, 0) == 6
    assert video_index(3, 30, -6) == 6
    # 算出负数时用第 0 帧
    assert video_index(1, 30, -6) == 0


def test_load_ped_labels():
    df = pd.DataFrame({
        "videoName": ["v", "v", "v"],
        "frameIndex": [0, 0, 2],
        "id": ["7", "8", "7"],
        "category": ["pedestrian", "car", "pedestrian"],
        "x1": [1.0, 0, 3.0], "y1": [1.0, 0, 3.0], "x2": [2.0, 0, 4.0], "y2": [2.0, 0, 4.0],
        "crowd": [False, False, True],
    })
    labels = load_ped_labels(df, "v")
    # 帧号从 1 开始，中间没人的第 2 帧也要有
    assert sorted(labels) == [1, 2, 3]
    assert labels[1] == [(7, (1.0, 1.0, 2.0, 2.0), False)]
    assert labels[2] == []
    assert labels[3][0][2] is True
