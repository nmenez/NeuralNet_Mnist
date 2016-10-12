import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    images = [image.reshape([784, 1])/255 for image in images]
    y = [vectorize(label) for label in labels]

    out_data = [(X, Y) for X, Y in zip(images, y)]

    return out_data, labels

def vectorize(y):
	out_y = np.zeros((10,1))
	out_y[y] = 1
	return out_y