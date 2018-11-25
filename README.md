# Digit Recognition

## Prerequisites
1. Install dependencies using pip:
   ```
   pip install -r requirements.txt
   ```
2. The `data/` directory includes the raw training and test datasets from [MNIST](http://yann.lecun.com/exdb/mnist/). They should be preprocessed into JSON files with:
   ```
   python preprocess.py data/train-images-idx3-ubyte.gz data/train-labels-idx1-ubyte.gz data/train.json
   python preprocess.py data/t10k-images-idx3-ubyte.gz data/t10k-labels-idx1-ubyte.gz data/test.json
   ```
