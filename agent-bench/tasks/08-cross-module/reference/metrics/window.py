class Window:
    """Keeps the last `size` samples."""

    def __init__(self, size):
        self.size = size
        self.samples = []

    def push(self, x):
        self.samples.append(x)
        if len(self.samples) > self.size:
            self.samples.pop(0)

    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0
