import gzip
import pickle as cPickle
import numpy as np
import tensorflow as tf
"""
Adaptar la red para el conjunto de imágenes MNIST. Dividir el conjunto de datos en un
70% para entrenamiento, 15% para validación y 15% para test. Realizar diferentes
configuraciones de la red para probar su rendimiento.

"""


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb') #interfaz para comprimir y descomprimir archivos
train_set, valid_set, test_set = cPickle.load(f, encoding='iso-8859-1')
f.close()

train_x, train_y = train_set


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print (train_y[57])


# TODO: the neural net!!
