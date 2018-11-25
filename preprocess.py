import gzip
import struct
import sys
from tqdm import tqdm

from digits import DigitData, DigitDatum

# http://yann.lecun.com/exdb/mnist/
def parse_idx(idx_file, progress=False):
    with gzip.open(idx_file, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        data_type = magic_number / 256
        num_dims = magic_number % 256
        assert data_type == 8 # unsigned byte
        dims = [struct.unpack('>I', f.read(4))[0] for _ in xrange(num_dims)]
        return read_array(f, dims, progress=True)

def read_array(f, dims, progress=False):
    rng = tqdm(xrange(dims[0])) if progress else xrange(dims[0])
    if len(dims) == 1:
        return [struct.unpack('>B', f.read(1))[0] for _ in rng]
    else:
        return [read_array(f, dims[1:]) for _ in rng]

def print_array(arr):
    for row in arr:
        print(''.join(['#' if c >= 128 else ' ' for c in row]))

# python preprocess.py images_file labels_file data/train.json
if __name__ == '__main__':
    data_in = sys.argv[1]
    labels_in = sys.argv[2]
    data_out = sys.argv[3]

    images = parse_idx(data_in, progress=True)
    labels = parse_idx(labels_in)
    data = DigitData([DigitDatum(image, label) for image, label in zip(images, labels)])
    print('Saving to %s' % data_out)
    data.save(data_out)
