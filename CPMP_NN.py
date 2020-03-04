from __future__ import absolute_import, division, print_function, unicode_literals
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import data as dt

from tensorflow import keras
import numpy as np


# EL PARAMETRO 'opt' INDICA CON UN 1 SI ES QUE QUIERE CARGAR LOS DATOS ENTRENADOS POR LA RED
# EN LA RUTA DECLARADA EN LA LINEA 28 CON LA VARIABLE 'check_point_path', CON UN VALOR DISTINTO A 1 NO SE CARGARA
def load_network(input_dim, output_dim, *opt):
    # CREANDO Y COMPILANDO EL MODELO DE LA RED
    clf = keras.Sequential([
        keras.layers.Flatten(input_shape=input_dim),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(output_dim, activation='softmax')
    ])
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # CARGANDO DATOS DE ENTRENAMIENTO
    if opt == 1:
        checkpoint_path = "data/NN_weights/nn_training_data.ckpt"
        clf.load_weights(checkpoint_path)
    return clf


# METODO PARA CARGAR LOS DATOS DE ENTRENAMIENTO DE LA RED
def load_data():
    test_path = "data/states/test.txt"
    test_labels_path = "data/states/testlabel.txt"
    train_path = "data/states/train.txt"
    train_labels_path = "data/states/trainlabel.txt"
    train_data, train_labels, c = dt.import_data(train_path, train_labels_path)
    test_data, test_labels, f = dt.import_data(test_path, test_labels_path)
    input_dim = train_data[0].shape[0]
    n = c*f
    output_dim = (n-1)*n
    return np.array(train_data, dtype=int), np.array(train_labels, dtype=int), np.array(test_data, dtype=int), \
           np.array(test_labels, dtype=int), input_dim, output_dim


# METODO PARA CARGAR LOS DATOS DE ENTRENAMIENTO DE LA RED CON ATRIBUTOS ESPECIFICOS
# 'opt' REPRESENTA EL STRING "0000" DONDE SE RELLENA CON 1 LOS ATRIBUTOS DESEADOS. ESTOS FUERON DESCRITOS EN EL ARCHIVO
# data.py
def load_filter_data(opt):
    test_path = "data/states/test/test_f.txt"
    test_labels_path = "data/states/test/testlabel_f.txt"
    train_path = "data/states/test/train_f.txt"
    train_labels_path = "data/states/test/trainlabel_f.txt"
    train_data, train_labels, c, f = dt.import_data(train_path, train_labels_path)
    test_data, test_labels, c, f = dt.import_data(test_path, test_labels_path)

    train_data, input_dim, output_dim = dt.filter_data(train_data, c, f, opt)
    test_data, input_dim, output_dim = dt.filter_data(test_data, c, f, opt)
    return np.array(train_data, dtype=int), np.array(train_labels, dtype=int), np.array(test_data, dtype=int), \
        np.array(test_labels, dtype=int), input_dim, output_dim


# SE ENTRENA LA RED NEURONAL, DEFINIDA CON EL MODELO EN EL METODO "load_network()"
def train_network(opt):
    # DEFINIR RUTA DE SALVACION DE DATOS
    checkpoint_path = "data/NN_weights/nn_training_data.ckpt"

    # SE CREA UN PUNTO DE GUARDADO PARA EL ENTRENAMIENTO DE LA RED
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # CARGANDO DATOS DE ENTRENAMIENTO SIN FILTRAR
    #train_data, train_labels, test_data, test_labels, input_dim, output_dim = load_data()

    # CARGANDO DATOS DE ENTRENAMIENTO FILTRADOS
    # MODIFICAR PARAMETRO STRING PARA ESCOGER LAS OPCIONES. MAYOR INFORMACION EN EL ARCHIVO data.py
    train_data, train_labels, test_data, test_labels, input_dim, output_dim = load_filter_data(opt)

    # CARGAR RED
    clf = load_network(input_dim, output_dim)

    # SE ENTRENA A LA RED
    clf.fit(train_data, train_labels, epochs=150, callbacks=[cp_callback], validation_data=(test_data, test_labels))

    # EVALUA EL ENTRENAMIENTO
    test_loss, test_acc = clf.evaluate(test_data, test_labels, verbose=2)

    # GUARDA LOS VALORES DE LA RED EN EL FORMATO 'checkpoint_path'
    clf.save_weights(checkpoint_path.format(epoch=0))

    # SE HACE UNA PRUEBA DE PRECISION A LA RED RESPECTO AL CONJUNTO DE PRUEBA DE DATOS
    print('\nTest accuracy1:', test_acc)
    test_loss, test_acc = clf.evaluate(test_data, test_labels, verbose=2)
    print('\nTest accuracy2:', test_acc)


