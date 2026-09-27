from .window import Window


def weekly_latency(samples, window=7):
    w = Window(window)
    for s in samples:
        w.push(s)
    return round(w.mean(), 2)
