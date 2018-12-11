import sys

from digits import DigitData
from model import NetworkModel

if __name__ == '__main__':
    model_file = sys.argv[1]
    test_data = DigitData.from_json(sys.argv[2])
    model = NetworkModel.load(model_file)
    print('Accuracy: %s' % model.evaluate(test_data, progress=True))
