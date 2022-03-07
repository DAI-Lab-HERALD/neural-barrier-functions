

class Partitions:
    def __init__(self, bounds):
        self.lower, self.upper = bounds

    @property
    def volume(self):
        return self.width.prod(dim=-1)

    @property
    def width(self):
        return self.upper - self.lower


class Partitioning:
    def __init__(self, initial, safe, unsafe, state_space):
        self.initial = Partitions(initial)
        self.safe = Partitions(safe)
        self.unsafe = Partitions(unsafe)
        self.state_space = Partitions(state_space) if state_space is not None else None
