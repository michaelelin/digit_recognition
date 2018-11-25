import math

class Perceptron:
    def __init__(self, layer, inputs):
        self.layer = layer
        self.weights = { feature: 0.0 for feature in inputs }

    def feed_forward(self, example):
        """
        Take inner product of weights and example, pass through activation function
        """
        weighted_sum = 0
        for name, value in example.items():
            weighted_sum += self.weights[name]*value

        return linear_rectify(weighted_sum)


def linear_rectify(val):
    return val if (val > 0) else 0


def sigmoid(val):
    return 1 / (1 + math.exp(-val))


class PerceptronLayer:
    def __init__(self, labels, inputs):
        self.size = size
        self.nodes = { label: Perceptron(self, inputs) for label in labels }

    def train(self, example, label):
        """
        example is a dict mapping feature name -> value

        Classify example
        If we guessed the correct label, do nothing
        If we got it wrong, change the weights
        """
        pass

    def feed_forward(self, example):
        return { label: node.feed_forward(example) for label, node in self.nodes.iteritems() }

    def classify(self, example):
        """
        Runs example through each node, returns the index of the perceptron that maximizes
        confidence
        """
        best_label = None
        best_value = None
        for label, value in self.feed_forward(example).iteritems():
            if best_value is None or value > best_value:
                best_label = label
        return best_label
