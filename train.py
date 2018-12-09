import sys
from perceptron import PerceptronLayer
from digits import DigitData
from tqdm import tqdm

EPOCHS = 10

def train(layer, train_data, test_data, epochs=EPOCHS):
    print('Initial accuracy: %s' % layer.evaluate(test_data, progress=True))
    for i in range(epochs):
        print('Epoch %s' % i)
        train_data.shuffle()
        for datum in tqdm(train_data.data[:10000]):
            layer.train(datum.features(), datum.label)
        print('Evaluating...')
        print('Accuracy: %s' % layer.evaluate(test_data, progress=True))



# python train.py data/train.json data/weights.json
if __name__ == '__main__':
    train_data = DigitData.from_json(sys.argv[1])
    test_data = DigitData.from_json(sys.argv[2])
    model_file = sys.argv[3]
    layer = PerceptronLayer(train_data.labels(), train_data.features())
    try:
        train(layer, train_data, test_data)
    except KeyboardInterrupt:
        layer.save(model_file)
    # write weights to sys.argv[2]

