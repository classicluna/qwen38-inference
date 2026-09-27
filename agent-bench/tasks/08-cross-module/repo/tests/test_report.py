from metrics.report import weekly_latency


def test_weekly():
    assert weekly_latency([100, 1, 2, 3, 4, 5, 6, 7]) == 4.0
