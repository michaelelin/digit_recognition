import sys

from digits import DigitData
from perceptron import PerceptronLayer

if __name__ == '__main__':
    model_file = sys.argv[1]
    test_data = DigitData.from_json(sys.argv[2])
    layer = PerceptronLayer.load(model_file)
    print('Accuracy: %s' % layer.evaluate(test_data, progress=True, threads=4))
