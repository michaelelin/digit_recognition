import json
import random

import activation
from tqdm import tqdm
from util import Vector

# the maximum value for tau. from what I understand, we're supposed to play around with
# this value and find the constant that works best.
C = 0.004

class Perceptron:
    def __init__(self, layer, num_weights, activation_fn):
        self.layer = layer
        self.weights = Vector(random.gauss(0, 0.001) for _ in xrange(num_weights))
        self.bias = random.gauss(0, 0.001)
        self.activation_fn = activation_fn

    def feed_forward(self, example):
        """
        Take inner product of weights and example, pass through activation function
        """
        return self.activation_fn.calculate((self.weights.dot(example)) + self.bias)

class PerceptronLayer:
    def __init__(self, num_labels, num_weights, activation_fn=activation.ReLU):
        self.nodes = [Perceptron(self, num_weights, activation_fn)
                      for _ in range(num_labels)]
        self.activation_fn = activation_fn

    def train(self, example, label):
        observed_label = self.classify(example)

        if label != observed_label:
            expected_perceptron = self.nodes[label]
            observed_perceptron = self.nodes[observed_label]
            tau = self.learning_rate(observed_perceptron.weights, expected_perceptron.weights, example)

            expected_perceptron.weights += example * tau
            observed_perceptron.weights -= example * tau

            expected_perceptron.bias += tau
            observed_perceptron.bias -= tau

    def evaluate(self, data, progress=False, threads=1):
        correct = 0

        for datum in (tqdm(data) if progress else data):
            if self.classify(datum.features()) == datum.label:
                correct += 1
        return float(correct) / len(data)

    def learning_rate(self, observed_weights, expected_weights, features):
        return min(((observed_weights - expected_weights).dot(features) + 1)
                   / (features.dot(features) * 2), C)

    def feed_forward(self, example):
        return Vector(node.feed_forward(example) for node in self.nodes)

    def classify(self, example):
        """
        Runs example through each node, returns the index of the perceptron that maximizes
        confidence
        """
        best_label = None
        best_value = None
        for label, value in enumerate(self.feed_forward(example)):
            if best_value is None or value > best_value:
                best_label = label
                best_value = value
        return best_label

    def save(self, filename):
        print('Saving weights to %s' % filename)
        serialized = [
            {
                'weights': perceptron.weights,
                'bias': perceptron.bias,
            } for perceptron in self.nodes
        ]
        with open(filename, 'w') as f:
            json.dump(serialized, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            serialized = json.load(f)
        num_labels = len(serialized)
        num_features = len(serialized[0]['weights'])
        layer = PerceptronLayer(num_labels, num_features)
        for ser_node, node in zip(serialized, layer.nodes):
            node.bias = ser_node['bias']
            node.weights = ser_node['weights']
        return layer
