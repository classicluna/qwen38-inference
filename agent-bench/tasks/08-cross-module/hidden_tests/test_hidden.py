from metrics.window import Window
from metrics.report import weekly_latency


def test_window_keeps_latest():
    w = Window(3)
    for x in range(10):
        w.push(x)
    assert w.samples == [7, 8, 9]


def test_report():
    assert weekly_latency([100, 1, 2, 3, 4, 5, 6, 7]) == 4.0
    assert weekly_latency([5]) == 5.0
