from .core import calc_total


def render(lines, tax_rate=0.0):
    return f"TOTAL {calc_total(lines, tax_rate):.2f}"
