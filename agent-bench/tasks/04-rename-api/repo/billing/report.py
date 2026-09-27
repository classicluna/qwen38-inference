from . import core


def monthly(invoices):
    return sum(core.calc_total(lines) for lines in invoices)
