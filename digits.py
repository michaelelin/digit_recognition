import json

class DigitData:
    def __init__(self, data):
        """
        data is a list of DigitDatum
        """
        pass

    @staticmethod
    def from_idx(data_file, label_file):
        """
        Returns a DigitData instance
        """
        pass

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
        with open(json_file, 'r') as f:
            data = json.load(f)
        return DigitData([DigitDatum.from_json(obj) for obj in data])

    def save(self, json_file):
        data = [datum.to_json() for datum in self.data]
        with open(json_file, 'w') as f:
            json.dump(data, f)





class DigitDatum:
    def __init__(self, pixels, label):
        pass

    def features(self):
        """
        Returns a dict from features names to values:
        - One features for each pixel (e.g. (4,3) -> 1)
        - components=1
        - components=2
        - components=3
        - Other features...
        """
        pass

    @staticmethod
    def from_json(datum):
        return DigitDatum(datum['pixels'], datum['label'])

    def to_json(self):
        return {
            'pixels': self.pixels,
            'label': self.label,
        }

