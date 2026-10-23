from sim1d import make_truth


def test_make_truth():
    assert make_truth(0, 2, 0.5, 5) == [0, 1, 2, 3, 4]
