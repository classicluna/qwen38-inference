from .core import compute_invoice_total


def render(lines, tax_rate=0.0):
    return f"TOTAL {compute_invoice_total(lines, tax_rate):.2f}"
