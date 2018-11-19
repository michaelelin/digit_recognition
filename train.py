import random
import sys
from perceptron import PerceptronLayer
from digits import DigitData

EPOCHS = 10

def train(layer, training_data, epochs=EPOCHS):
    for i in range(epochs):
        print('Epoch %s' % i)
        random.shuffle(training_data)
        for datum in training_data:
            layer.train(datum.features(), datum.label)



# python train.py data/train.json data/weights.json
if __name__ == '__main__':
    data = DigitData.from_json(sys.argv[1])
    layer = PerceptronLayer(data.features, data.labels)
    train(layer, data)
    # write weights to sys.argv[2]

