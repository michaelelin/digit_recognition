import sys
from tqdm import tqdm

import activation
from digits import DigitData
from model import NetworkModel

EPOCHS = 60

def train(model, train_data, test_data, model_file, epochs=EPOCHS):
    try:
        for i in range(epochs):
            print('Epoch %s' % i)
            train_data.shuffle()
            for datum in tqdm(train_data.data[:10000]):
                model.train(datum.features(), datum.label_vec)
            print('Evaluating...')
            print('Accuracy: %s' % model.evaluate(test_data, progress=True))
    except KeyboardInterrupt:
        pass
    finally:
        model.save(model_file)


# python train.py data/train.json data/weights.json
if __name__ == '__main__':
    train_data = DigitData.from_json(sys.argv[1])
    test_data = DigitData.from_json(sys.argv[2])
    model_file = sys.argv[3]
    #layer = PerceptronLayer(train_data.num_labels(), train_data.num_features())
    model = NetworkModel(train_data.num_features(), 30, 30, train_data.num_labels(),
                         activation_fns=[activation.ReLU, activation.ReLU, activation.Sigmoid])
    train(model, train_data, test_data, model_file)
    # write weights to sys.argv[2]

