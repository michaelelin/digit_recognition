import math

class ActivationFunction:
    @classmethod
    def name(cls):
        return cls.NAME

class Linear(ActivationFunction):
    NAME = 'linear'
    @staticmethod
    def calculate(val):
        return val

    @staticmethod
    def derivative(val):
        return 1.0

class ReLU(ActivationFunction):
    NAME = 'relu'
    @staticmethod
    def calculate(val):
        return max(0.0, val)

    @staticmethod
    def derivative(val):
        if val > 0:
            return 1.0
        else:
            return 0.0

class Sigmoid(ActivationFunction):
    NAME = 'sigmoid'
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


ACTIVATION_FUNCTIONS = [Linear, ReLU, Sigmoid]
def from_name(name):
    for fn in ACTIVATION_FUNCTIONS:
        if fn.name() == name:
            return fn
