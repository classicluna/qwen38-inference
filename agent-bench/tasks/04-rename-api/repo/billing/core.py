def calc_total(lines, tax_rate=0.0):
    subtotal = sum(q * p for q, p in lines)
    return round(subtotal * (1 + tax_rate), 2)
