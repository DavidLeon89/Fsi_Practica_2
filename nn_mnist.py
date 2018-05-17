import gzip
import pickle as cPickle
import numpy as np
import tensorflow as tf
import sys
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

#ENTRENAMIENTO
train_x, train_y = train_set
train_y = one_hot(train_y.astype(int), 10)

#VALIDAR
valid_x, valid_y = valid_set
valid_y = one_hot(valid_y.astype(int), 10)

#TEST
test_x, test_y = test_set
test_y = one_hot(test_y.astype(int), 10)

# PLACEHOLDER Inserta un marcador de posición para un tensor que siempre se alimentará.
x = tf.placeholder("float", [None, 784])  # samples 784 entradas
y_ = tf.placeholder("float", [None, 10])  # labels de tamaño 10


## esto es para la formula del sismode que vimos en clase sumatorio(x*w)+b
W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1) #784 entras 28x28 tamaño imagen
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session() #crear una sesicon y ejecutar el grafo
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")


#def imprimir(x_d, y_d):
batch_size = 20
valOK = 0
valEr = 0
testOK = 0
testEr = 0

graficaEntreno = []
graficaErroresEntreno = []
graficaValidacion = []
graficaErroresValidacion = []
errorActual = 0
epoch = 0
tanMaximo = sys.maxsize

print("-----------------Entrenamiento y validacion iniciados---------------------")
#La condición de parada de la red se hará mediante la comparación porcentual del error actual con el anterior.
#Es decir, si la diferencia es menor que un cierto umbral se parará.
while errorActual > 0.001*tanMaximo + tanMaximo or errorActual < tanMaximo - 0.001*tanMaximo:
    tanMaximo = errorActual
    for jj in range(len(train_x) // batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    #print("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
    errorActual = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    graficaErroresEntreno = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}) #necesito un vector
    graficaEntreno.append(graficaErroresEntreno)
    print("Epoch #:", epoch, "Error: ", graficaErroresEntreno)

    graficaErroresValidacion = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})  # necesito un vector
    graficaValidacion.append(graficaErroresValidacion)
    print("Epoch #:", epoch, "Error: ", graficaErroresValidacion)

    epoch += 1

resultado = sess.run(y, feed_dict={x: test_x})

for b, r in zip(test_y, resultado):
    if np.argmax(b) == np.argmax(r):
        valOK += 1
        print(b, "-->", r, "Acierto número ", valOK)

    else:
        print(b, "-->", r, "Error número ", valEr)
        valEr += 1

print("-----------------Entrenamiento y validacion Finalizados---------------------")

print("-----------TEST--------------------")
##"Epoch #: Test", "Error: ", sess.run(loss, feed_dict={x: x_test, y_: y_test})
resultadoTest = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, resultadoTest):
    if np.argmax(b) == np.argmax(r):
        testOK += 1
        print(b, "-->", r, "Acierto número ", testOK)

    else:
        testEr += 1
        print(b, "-->", r, "Error número ", testEr)

print()
print("----------------------------------------------------------------------------------")
print("He encontrado ", valOK, " aciertos en la validacion y ", valEr,
      " errores. Porcentaje de acierto", int(valOK/(valOK+valEr)*100), "%")
print("He encontrado ", testOK, " aciertos en los test y ", testEr,
      " errores. Porcentaje de acierto", int(testOK/(testOK+testEr)*100), "%")






# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt
#voy a mostrar los errores del entreno
plt.title("Entreno")
plt.plot(graficaEntreno)
plt.show()


plt.title("Validacion")
plt.plot(graficaValidacion)
plt.show()

"""
plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print (train_y[57])
"""

# TODO: the neural net!!
