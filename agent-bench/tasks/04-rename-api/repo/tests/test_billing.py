from billing import calc_total
from billing.invoice import render


def test_total():
    assert calc_total([(2, 1.5)], 0.1) == 3.3


def test_render():
    assert render([(1, 2.0)]) == "TOTAL 2.00"
