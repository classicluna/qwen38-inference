from . import core


def monthly(invoices):
    return sum(core.compute_invoice_total(lines) for lines in invoices)
