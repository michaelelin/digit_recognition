class Vector(list):
    def __add__(self, other):
        return Vector(a + b for a, b in zip(self, other))

    def __iadd__(self, other):
        for i, b in enumerate(other):
            self[i] += b
        return self

    def __sub__(self, other):
        return Vector(a - b for a, b in zip(self, other))

    def __isub__(self, other):
        for i, b in enumerate(other):
            self[i] -= b
        return self

    def __mul__(self, other):
        if isinstance(other, list): # elementwise multiplication
            return Vector(a * b for a, b in zip(self, other))
        else:
            return Vector(a * other for a in self)

    def dot(self, other):
            return sum(a * b for a, b in zip(self, other))

    def argmax(self):
        return max(range(len(self)), key=lambda i: self[i])

    @classmethod
    def zeros(cls, size):
        return cls(0.0 for _ in xrange(size))
