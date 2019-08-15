"""Kannada-MNIST dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

from ..utils.data_utils import get_file
import numpy as np


def load_data():
    """Loads the Kannada-MNIST dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'kannada-mnist')
    base = 'https://github.com/vinayprabhu/Kannada_MNIST/raw/master/data/output_tensors/MNIST_format/'

    files = [
        ('f1f06031670554894b697988b906b036',  'y_kannada_MNIST_train-idx1-ubyte.gz'),
        ('db2cd247b479801b3a882ab980b340c3',  'X_kannada_MNIST_train-idx3-ubyte.gz'),
        ('b426d27d50a5fd8411b9e5838868be4b',  'y_kannada_MNIST_test-idx1-ubyte.gz'),
        ('8682876e8fbc8cd0e7bb07fc2f502335',  'X_kannada_MNIST_test-idx3-ubyte.gz'),
    ]

    paths = [
        get_file(fname, origin=base + fname, cache_subdir=dirname, file_hash=file_hash)
        for file_hash, fname in files
    ]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
