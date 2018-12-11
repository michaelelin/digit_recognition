from tqdm import tqdm

import activation
from perceptron import PerceptronLayer

LEARNING_RATE = 3.0

class NetworkModel:
    # NetworkModel(784, 10)
    # NetworkModel(784, 30, 10)
    def __init__(self, *node_counts):
        self.layers = [PerceptronLayer(n2, n1, activation.Sigmoid) for n1, n2 in zip(node_counts, node_counts[1:])]

    def feed_forward(self, example):
        outputs = example
        for layer in self.layers:
            outputs = layer.feed_forward(outputs)
        return outputs

    def train(self, example, expected):
        outputs = [example]
        for layer in self.layers:
            outputs.append(layer.feed_forward(outputs[-1]))
        error = expected - outputs[-1]
        modified_error = [e * node.activation_fn.derivative(node.weights.dot(outputs[-2]))
                          for e, node in zip(error, self.layers[-1].nodes)]
        for k, node in enumerate(self.layers[-1].nodes):
            node.weights += (outputs[-2] * modified_error[k]) * LEARNING_RATE
            node.bias += modified_error[k] * LEARNING_RATE

    def classify(self, example):
        return self.feed_forward(example).argmax()

    def evaluate(self, data, progress=False):
        correct = 0

        for datum in (tqdm(data) if progress else data):
            if self.classify(datum.features()) == datum.label:
                correct += 1
        return float(correct) / len(data)
