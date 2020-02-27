import data as dt
import numpy as np
import CPMP as cpmp
import copy

from tensorflow import keras

import CPMP_NN as nn


def generate_movements_array(columns):
    movements_set = np.array([])
    for i in range(columns):
        for j in i:
            if i == j:
                continue
            movements_set = np.append(movements_set, (i, j))
    return movements_set


def recognize_column_movement(movements_set, movement):
    return np.where(movements_set == movement)


def recognize_movement(movements_set, movement):
    return movements_set[movement]


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



def norm_eval(array, f):
    for i in range(len(array) - 1):
        if (i+1) % f == 0:
            continue
        if array[i] < array[i+1]:
            return False
    return True


# ESTE METODO ENTREGA LA CANTIDAD DE PASOS DE UNA SOLUCION OPTIMA
def best_solution_depth(s):
    node = cpmp.CPMP_Node(s)
    node = cpmp.dlts_lds(node)
    return node.contar_profundidad()


def transform_state(state, opt, last_move):
    arr = cpmp.normalize_array(state.transform_to_array())
    cp = copy.deepcopy(arr)
    for x in opt:
        if x == 0 and opt[x] == "1":
            arr = np.append(arr, state.group_values_array())
            continue
        if x == 1 and opt[x] == "1":
            arr = np.append(arr, state.get_base_differences(cp))
            continue
        if x == 2 and opt[x] == "1":
            arr = np.append(arr, state.get_top_differences(cp))
            continue
        if x == 3 and opt[x] == "1":
            arr = np.append(arr, state.get_ba(cp))
    arr = np.append(arr, last_move)
    arr = np.expand_dims(arr, last_move)
    return arr


# 'free_moves' REPRESENTA LA CANTIDAD DE PASOS EN QUE SE LLEGO A LA SOLUCION OPTIMA
# 'n' ES EL MULTIPLICADOR DE 'free_moves' QUE REPRESENTA LA CANTIDAD DE MOVIMIENTOS MAXIMA PARA PODER RESOLVER EL ESTADO
def nn_solver(free_moves, state, clf, columns, row):
    movements = generate_movements_array(columns)
    n = 3
    # SE TRANSFORMA EL ESTADO
    arr_state = transform_state(state, "1111", 0)
    # CUENTA LA CANTIDAD DE MOVIMIENTOS QUE SE HICIERON PARA RESOLVER EL PROBLEMA
    count = 0
    lastmove = len(movements)
    # SE INTENTA RESOLVER EL ESTADO EN 3*FREE_MOVES
    for i in range(n*free_moves):
        j = len(movements)-1
        count = count + 1
        # GENERO UN PASO CON LA RED
        predict = clf.predict(arr_state)
        predicted_step = predict.argsort()
        # SE HACE EL MOVIMIENTO PRODUCIDO POR LA RED
        # EN CASO DE QUE ESTE NO SEA VALIDO, GENERO EL SIGUIENTE PASO QUE PREDIJO LA RED
        while not movement(predicted_step[0][j], state):
            j = j - 1
            if opuesto(lastmove) == predicted_step[0][j]:
                j = j - 1
        lastmove = predicted_step[0][j]
        arr_state = transform_state(state, "1111", lastmove)
        if state.eval_state():
            return 1, count
    return 0, 0


# 'n' REPRESENTA LA CANTIDAD DE ESTADOS PARA PROBAR A LA RED
def nn_test(n):
    opt = "1111"
    columns, rows = dt.get_state_dims("data/states/test.txt")
    input_dim, output_dim = dt.get_nn_dims(columns, rows, opt)
    clf = nn.load_network(input_dim, output_dim, opt)
    total, success = 0, 0
    for x in range(n):
        total = total + 1
        s = cpmp.init(3, 5, 2)
        depth = best_solution_depth(s)
        y = nn_solver(depth, s, clf, columns, rows)
        if y == 1:
            success = success + 1
    return (success / total) * 100






