import json
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
        inputs = [example]
        for layer in self.layers:
            inputs.append(layer.feed_forward(inputs[-1]))
        error = expected - inputs[-1]
        modified_errors = [None] * len(self.layers)
        modified_errors[-1] = [e * self.layers[-1].activation_fn.derivative(node.weights.dot(inputs[-2]) +
                                                                 node.bias)
                               for e, node in zip(error, self.layers[-1].nodes)]

        for i in xrange(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            modified_errors[i] = [(layer.activation_fn.derivative(node.weights.dot(inputs[i]) +
                                                                           node.bias) *
                                   sum(next_node.weights[j] * modified_errors[i+1][k]
                                       for k, next_node in enumerate(self.layers[i+1].nodes)))
                                  for j, node in enumerate(layer.nodes)]

        for i, layer in enumerate(self.layers):
            for j, node in enumerate(layer.nodes):
                node.weights += inputs[i] * (modified_errors[i][j] * LEARNING_RATE)
                node.bias += modified_errors[i][j] * LEARNING_RATE
        # for k, node in enumerate(self.layers[-1].nodes):
            # node.weights += (inputs[-1] * modified_errors[-1][k]) * LEARNING_RATE
            # node.bias += modified_errors[-1][k] * LEARNING_RATE



    def classify(self, example):
        return self.feed_forward(example).argmax()

    def evaluate(self, data, progress=False):
        correct = 0

        for datum in (tqdm(data) if progress else data):
            if self.classify(datum.features()) == datum.label:
                correct += 1
        return float(correct) / len(data)

    def save(self, filename):
        print('Saving weights to %s' % filename)
        serialized = [layer.serialize() for layer in self.layers]
        with open(filename, 'w') as f:
            json.dump(serialized, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            serialized = json.load(f)
        layers = [PerceptronLayer.deserialize(layer_ser) for layer_ser in serialized]
        model = NetworkModel()
        model.layers = layers
        return model
