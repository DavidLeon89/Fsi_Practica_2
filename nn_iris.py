import tensorflow as tf
import numpy as np


# Traducir una lista de etiquetas en una matriz de 0 y una 1.
# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    # funcion de tf x= indice, n profundidad
    """
    :param x: label (int) etiquetas
    :param n: number of bits indices
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)  # si es tipo lista lo pasara a array
    x = x.flatten()  # pasa matriz a array
    o_h = np.zeros((len(x), n))  # matriz llena de ceros
    o_h[np.arange(len(x)), x] = 1  # arange: genera el rango de lengX a X
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data ¡coctel!
x_data = data[:, 0:4].astype('f4')  # las muestras son las cuatro primeras filas de datos
y_data = one_hot(data[:, 4].astype(int),
                 3)  # las etiquetas están en la última fila. Luego los codificamos en one hot code

"""Dividir el conjunto de datos en un 70% para entrenamiento, 15% para validación y 15%
para test. Realizar el entrenamiento mientras se observa el proceso sobre el conjunto
de validación. Una vez finalizado el entrenamiento, comprobar el resultado sobre el
conjunto de test."""

# Dividir el conjunto de datos en un 70% para entrenamiento de 0 a 105
x_entrenamiento = data[0:105, 0:4].astype('f4')  # las muestras son las cuatro primeras filas de datos
y_entrenamiento = one_hot(data[0:105, 4].astype(int),
                          3)  # las etiquetas están en la última fila. Luego los codificamos en one hot code

# 15% para validación de 106 a 128,5... nos quedamos con enteros (22,5)
x_validacion = data[106:128, 0:4].astype('f4')  # las muestras son las cuatro primeras filas de datos
y_validacion = one_hot(data[106:128, 4].astype(int),
                       3)  # las etiquetas están en la última fila. Luego los codificamos en one hot code

# 15% para test de 129 a 128,5... nos quedamos con enteros 22,5
x_test = data[129:150, 0:4].astype('f4')  # las muestras son las cuatro primeras filas de datos
y_test = one_hot(data[129:150, 4].astype(int),
                 3)  # las etiquetas están en la última fila. Luego los codificamos en one hot code

"""print ("\nSome samples...")
for i in range(20):
    print (x_data[i], " -> ", y_data[i])
print"""

# PLACEHOLDER Inserta un marcador de posición para un tensor que siempre se alimentará.
x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

## esto es para la formula del sismode que vimos en clase sumatorio(x*w)+b
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
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
print("-----------------Entrenamiento y validacion iniciados---------------------")
for epoch in range(100):
    for jj in range(len(x_entrenamiento) // batch_size):
        batch_xs = x_entrenamiento[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_entrenamiento[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    print("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
    resultadoValidacion = sess.run(y, feed_dict={x: x_validacion})  # ¿que esta haciendo aqui?
for b, r in zip(y_validacion, resultadoValidacion):
    if np.argmax(b) == np.argmax(r):
        valOK += 1
        print(b, "-->", r, "Acierto número ", valOK)

    else:
        print(b, "-->", r, "Error número ", valEr)
        valEr += 1

print("-----------------Entrenamiento y validacion Finalizados---------------------")

print("-----------TEST--------------------")
##"Epoch #: Test", "Error: ", sess.run(loss, feed_dict={x: x_test, y_: y_test})
resultadoTest = sess.run(y, feed_dict={x: x_test})
for b, r in zip(y_test, resultadoTest):
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



    # FIN DEL PRIMER FOR


# imprimir(x_data,y_data)

# Realizar el entrenamiento mientras se observa el proceso sobre el conjunto de validación.
#imprimir(x_validacion, y_validacion)
"""
Ejemplo del profesor
for epoch in xrange(100):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"








"""