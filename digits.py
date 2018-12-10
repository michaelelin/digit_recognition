import json
import itertools
import random

class DigitData:
    def __init__(self, data):
        """
        data is a list of DigitDatum
        """
        self.data = data

    @staticmethod
    def from_json(json_file):
        """
        [
            {
                'pixels': [[some matrix]],
                'label': 4
            }
            ...
        ]
        """
        print('Loading data from %s' % json_file)
        with open(json_file, 'r') as f:
            data = json.load(f)
        print('Loaded.')
        return DigitData([DigitDatum.from_json(obj) for obj in data])

    def num_labels(self):
        return len(set(datum.label for datum in self.data))

    def num_features(self):
        return len(self.data[0].features())

    def shuffle(self):
        random.shuffle(self.data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def save(self, json_file):
        data = [datum.to_json() for datum in self.data]
        with open(json_file, 'w') as f:
            json.dump(data, f)





class DigitDatum:
    def __init__(self, pixels, label):
        self.pixels = pixels
        self.label = label
        self._features = None
        self.features()

    def features(self):
        """
        Returns a dict from features names to values:
        - One features for each pixel (e.g. (4,3) -> 1)
        - components=1
        - components=2
        - components=3
        - Other features...
        """
        if not self._features:
            width = len(self.pixels[0])
            height =  len(self.pixels)
            self._features = [self.pixels[i / width][i % width] for i in xrange(width * height)]
        return self._features

    @staticmethod
    def from_json(datum):
        return DigitDatum(datum['pixels'], datum['label'])

    def to_json(self):
        return {
            'pixels': self.pixels,
            'label': self.label,
        }

    def print_digit(self):
        def format_value(val):
            if val > 128:
                return '#'
            elif val > 0:
                return '-'
            else:
                return ' '
        for row in self.pixels:
            print(''.join([format_value(val) for val in row]))

