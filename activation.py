import math

class Linear:
    @staticmethod
    def calculate(val):
        return val

    @staticmethod
    def derivative(val):
        return 1.0

class ReLU:
    @staticmethod
    def calculate(val):
        return max(0.0, val)

    @staticmethod
    def derivative(val):
        if val > 0:
            return 1.0
        else:
            return 0.0

class Sigmoid:
    @staticmethod
    def calculate(val):
        if val < -100.0:
            return 0.0
        return 1.0 / (1 + math.exp(-val))

    @staticmethod
    def derivative(val):
        if val < -100.0:
            return 0.0
        e = math.exp(-val)
        return e / ((1 + e) * (1 + e))
