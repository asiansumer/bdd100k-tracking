from pytest import approx

from box_utils import xyxy_to_xyah, xyah_to_xyxy


def test_xyxy_to_xyah():
    # 你刚才手算的例子
    assert xyxy_to_xyah((100, 200, 140, 300)) == approx([120, 250, 0.4, 100])


def test_round_trip():
    # 换过去再换回来，应该回到原来的框
    box = (100, 200, 140, 300)
    assert xyah_to_xyxy(xyxy_to_xyah(box)) == approx(list(box))
