import json
import math
from tqdm import tqdm
from util import subtract_vectors, dot_product

# the maximum value for tau. from what I understand, we're supposed to play around with
# this value and find the constant that works best.
C = 0.004

class Perceptron:
    def __init__(self, layer, inputs):
        self.layer = layer
        self.weights = { feature: 0.0 for feature in inputs }
        self.bias = 0.0

    def feed_forward(self, example):
        """
        Take inner product of weights and example, pass through activation function
        """
        weighted_sum = 0
        for name, value in example.items():
            weighted_sum += self.weights[name]*value
        weighted_sum += self.bias

        return linear_rectify(weighted_sum)


def linear_rectify(val):
    return val if (val > 0) else 0


def sigmoid(val):
    return 1 / (1 + math.exp(-val))


class PerceptronLayer:
    def __init__(self, labels, inputs):
        self.nodes = { label: Perceptron(self, inputs) for label in labels }

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

            for feature, value in example.items():
                expected_perceptron.weights[feature] += value * tau
                observed_perceptron.weights[feature] -= value * tau
            expected_perceptron.bias += tau
            observed_perceptron.bias -= tau

    def evaluate(self, data, progress=False):
        correct = 0

        for datum in (tqdm(data) if progress else data):
            if self.classify(datum.features()) == datum.label:
                correct += 1
        return float(correct) / len(data)

    def learning_rate(self, observed_weights, expected_weights, features):
        return min((dot_product(subtract_vectors(observed_weights, expected_weights), features) + 1) 
                    / (dot_product(features, features) * 2), C)

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
                best_value = value
        return best_label

    def save(self, filename):
        print('Saving weights to %s' % filename)
        serialized = {
            label: {
                'weights': { feature: value for feature, value in
                            perceptron.weights.iteritems() },
                'bias': perceptron.bias,
            } for label, perceptron in self.nodes.iteritems()
        }
        with open(filename, 'w') as f:
            json.dump(serialized, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            serialized = json.load(f)
        labels = map(int, serialized.keys())
        inputs = map(int, serialized[labels[0]]['weights'].keys())
        layer = PerceptronLayer(labels, inputs)
        for label, node in layer.nodes.iteritems():
            node.bias = serialized[label]['bias']
            node.weights = { int(feature): value for feature, value in
                            serialized[label]['weights'].iteritems() }
        return layer
