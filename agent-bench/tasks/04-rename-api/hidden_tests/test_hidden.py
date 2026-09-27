import pathlib, billing
from billing import compute_invoice_total
from billing.report import monthly


def test_new_name():
    assert compute_invoice_total([(2, 1.5)], 0.1) == 3.3
    assert monthly([[(1, 1.0)], [(2, 2.0)]]) == 5.0


def test_old_name_gone():
    assert not hasattr(billing, "calc_total")
    root = pathlib.Path(billing.__file__).parent.parent
    for p in list(root.glob("billing/*.py")) + list(root.glob("tests/*.py")):
        assert "calc_total" not in p.read_text(), p
