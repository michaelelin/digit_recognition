import sys
from digits import DigitData

# python preprocess.py images_file labels_file data/train.json
if __name__ == '__main__':
    data_in = sys.argv[1]
    labels_in = sys.argv[2]
    data_out = sys.argv[3]
    data = DigitData.from_idx(data_in, labels_in)
    data.save(data_out)
