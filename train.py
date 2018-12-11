import sys
from tqdm import tqdm

from digits import DigitData
from model import NetworkModel

EPOCHS = 10

def train(model, train_data, test_data, model_file, epochs=EPOCHS):
    print('Initial accuracy: %s' % model.evaluate(test_data, progress=True))
    try:
        for i in range(epochs):
            print('Epoch %s' % i)
            import ipdb; ipdb.set_trace()
            train_data.shuffle()
            for datum in tqdm(train_data.data[:10000]):
                model.train(datum.features(), datum.label_vec)
            print('Evaluating...')
            print('Accuracy: %s' % model.evaluate(test_data, progress=True))
    except KeyboardInterrupt:
        pass
    finally:
        #layer.save(model_file)
        pass



# python train.py data/train.json data/weights.json
if __name__ == '__main__':
    train_data = DigitData.from_json(sys.argv[1])
    test_data = DigitData.from_json(sys.argv[2])
    model_file = sys.argv[3]
    #layer = PerceptronLayer(train_data.num_labels(), train_data.num_features())
    model = NetworkModel(train_data.num_features(), train_data.num_labels())
    train(model, train_data, test_data, model_file)
    # write weights to sys.argv[2]

