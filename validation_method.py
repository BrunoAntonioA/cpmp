import os
import data as dt
import numpy as np
import CPMP as cpmp
import copy

from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mayor(n, predict):
    x = np.sort(predict)
    print("n: ", n)
    print("x[0]: ", x[0])
    value = x[0][n]
    index = np.where(predict == value)
    return index[0][0]


def validate_predict_data(predict_data, test_labels):
    length = len(predict_data)
    count = 0
    for i in range(length):
        if mayor(predict_data[i]) == test_labels[i]:
            count = count + 1

    print("Total de datos = ", length)
    print("Acertados = ", count)
    print("Porcentaje de acierto = ", count/length * 100, "%")

    return count/length


def generate_movement(move):
    if move == 0:
        return 0, 1
    if move == 1:
        return 0, 2
    if move == 2:
        return 1, 0
    if move == 3:
        return 1, 2
    if move == 4:
        return 2, 0
    if move == 5:
        return 2, 1


def opuesto(move):
    if move == 0:
        return 2
    if move == 1:
        return 4
    if move == 2:
        return 0
    if move == 3:
        return 5
    if move == 4:
        return 1
    if move == 5:
        return 3
    return 6


def movement(move, state):
    tpl = generate_movement(move)
    if not state.movement(tpl[0], tpl[1]):
        return 0
    return 1


def cargar_red(path, input_dim):

    checkpoint_path = path
    clf = keras.Sequential([
        keras.layers.Flatten(input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(6, activation='softmax')
    ])
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Loads the weights
    clf.load_weights(checkpoint_path)
    return clf


def norm_eval(array, f):
    for i in range(len(array) - 1):
        if (i+1) % f == 0:
            continue
        if array[i] < array[i+1]:
            return False
    return True


def nn_solution(n, state, clf, input_dim):
    x = state.transform_to_array()
    if input_dim == 19:
        x = np.append(x, state.group_values_array())
    x = np.append(x, 0)
    x = cpmp.normalize_array(x)
    x = (np.expand_dims(x, 0))
    count = 0
    lastmove = 6
    for i in range(3*n):
        j = 5
        count = count + 1
        predict = clf.predict(x)
        predict = np.array(predict)
        predi = predict.argsort()
        #state.print_yard()
        #print("predict: ", predict)
        #print("predi[j]: ", predi[0][j])
        #print("----------")
        if opuesto(lastmove) == predi[0][j]:
            j = j - 1
        while not movement(predi[0][j], state):
            j = j - 1
            if opuesto(lastmove) == predi[0][j]:
                j = j - 1
        lastmove = predi[0][j]
        x = state.transform_to_array()
        if input_dim == 19:
            x = np.append(x, state.group_values_array())
        x = np.append(x, lastmove)
        x = cpmp.normalize_array(x)
        x = (np.expand_dims(x, 0))
        if state.eval_state():
            #state.print_yard()
            #print("---------------------------------------------")
            return 1, count
    return 0, 0


def best_solution_depth(s):
    node = cpmp.CPMP_Node(s)
    node = cpmp.dlts_lds(node)
    return node.contar_profundidad()


def testing_NN(n, path, input_dim):
    clf = cargar_red(path, input_dim)
    aciertos = 0
    total = 0
    for x in range(n):
        total = total + 1
        s = cpmp.init(3, 5, 2)
        depth = best_solution_depth(s)
        y = nn_solution(depth, s, clf, input_dim)
        if y[0] == 1:
            aciertos = aciertos + 1
    print("termina ejecucion")
    return total, aciertos


r1 = testing_NN(20, "NN_weights/NN_normalize_gv/nn_19_gv.ckpt", 19)
r2 = testing_NN(20, "NN_weights/NN_normalize_gv/nn_16.ckpt", 16)

print("r1: ", r1)
print("r2: ", r2)





