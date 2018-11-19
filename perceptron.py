class Perceptron:
    def __init__(self, layer, inputs):
        self.layer = layer
        self.weights = { feature: 0.0 for feature in inputs }

    def feed_forward(self, example):
        """
        Take inner product of weights and example, pass through activation function
        """
        pass

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
        pass

    def classify(self, example):
        """
        Runs example through each node, returns the index of the perceptron that maximizes
        confidence
        """
        pass
