import json
import math
from tqdm import tqdm
from util import subtract_vectors, dot_product

# the maximum value for tau. from what I understand, we're supposed to play around with
# this value and find the constant that works best.
C = 0.004

class Perceptron:
    def __init__(self, layer, num_weights):
        self.layer = layer
        self.weights = [0.0] * num_weights
        self.bias = 0.0

    def feed_forward(self, example):
        """
        Take inner product of weights and example, pass through activation function
        """
        weighted_sum = 0
        for weight, value in zip(self.weights, example):
            weighted_sum += weight * value
        weighted_sum += self.bias

        return linear_rectify(weighted_sum)


def linear_rectify(val):
    return val if (val > 0) else 0


def sigmoid(val):
    return 1 / (1 + math.exp(-val))


class PerceptronLayer:
    def __init__(self, num_labels, num_weights):
        self.nodes = [Perceptron(self, num_weights) for _ in range(num_labels)]

    def train(self, example, label):
        """
        example is a dict mapping feature name -> value

        Classify example
        If we guessed the correct label, do nothing
        If we got it wrong, change the weights
        """
        observed_label = self.classify(example)

        if label != observed_label:
            expected_perceptron = self.nodes[label]
            observed_perceptron = self.nodes[observed_label]
            tau = self.learning_rate(observed_perceptron.weights, expected_perceptron.weights, example)

            for i, value in enumerate(example):
                expected_perceptron.weights[i] += value * tau
                observed_perceptron.weights[i] -= value * tau
            expected_perceptron.bias += tau
            observed_perceptron.bias -= tau

    def evaluate(self, data, progress=False, threads=1):
        correct = 0

        for datum in (tqdm(data) if progress else data):
            if self.classify(datum.features()) == datum.label:
                correct += 1
        return float(correct) / len(data)

    def learning_rate(self, observed_weights, expected_weights, features):
        return min((dot_product(subtract_vectors(observed_weights, expected_weights), features) + 1) 
                    / (dot_product(features, features) * 2), C)

    def feed_forward(self, example):
        return [node.feed_forward(example) for node in self.nodes]

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
