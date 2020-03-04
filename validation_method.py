
import numpy as np
import CPMP as cpmp
import copy

import data as dt
import CPMP_NN as nn


# METODO PARA GENERAR LOS POSIBLES MOVIMIENTOS PARA UN ESTADO DE 'columns' COLUMNAS
def generate_movements_array(columns):
    movements_set = []
    for i in range(columns):
        for j in range(columns):
            if i == j:
                continue
            movements_set.append((i, j))
    return movements_set


# METODO PARA TRANSFORMAR UN MOVIMIENTO EN FORMATO TUPLA A FORMATO DE UN ENTERO
def tuple_to_move(movements_set, tuple):
    for x in range(len(movements_set)):
        if movements_set[x] == tuple:
            return x


# METODO PARA TRANSFORMAR UN MOVIMIENTO EN FORMATO TUPLA A FORMATO DE UN ENTERO
def move_to_tuple(movements_set, mov):
    return movements_set[mov]


# GENERA UN MOVIMIENTO PARA UN ESTADO, EN CASO DE NO PODER REALIZARSE RETORNA 0
def movement(move, state, movements):
    tpl = move_to_tuple(movements, move)
    if not state.movement(tpl[0], tpl[1]):
        return 0
    return 1


# EVALUA UN ESTADO EN FORMATO DE ARRAY
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


def opuesto(movements, mov):
    tuple = move_to_tuple(movements, mov)
    for x in range(len(movements)):
        if movements[x] == (tuple[1], tuple[0]):
            opuesto = movements[x]
            mov_opuesto = tuple_to_move(movements, opuesto)
            return mov_opuesto


def index_mayor(mayor, arreglo):
    for i in range(len(arreglo)):
        if mayor == arreglo[i]:
            return i


# TRANSFORMA UN ESTADO EN FORMATO CPMP A UN ARRAY, AGREGA LOS ATRIBUTOS DE MANERA QUE PUEDA SER UTILIZADO POR LA RED
# EL PARAMETRO 'opt' ES LA OPCION PARA SABER QUE ATRIBUTOS SE DESEAN AGREGAR
# 'last_move' ES EL PASO DEL ESTADO DEL QUE VIENE. SE AGREGA COMO ATRIBUTO AL ESTADO
def transform_state(state, opt, last_move):
    arr = cpmp.normalize_array(state.transform_to_array())
    cp = copy.deepcopy(arr)
    for x in range(len(opt)):
        if x == 0 and opt[x] == "1":
            arr = np.append(arr, state.group_values_array())
            continue
        if x == 1 and opt[x] == "1":
            arr = np.append(arr, state.get_base_differences_from_normalize(cp))
            continue
        if x == 2 and opt[x] == "1":
            arr = np.append(arr, state.get_top_differences_from_normalize(cp))
            continue
        if x == 3 and opt[x] == "1":
            arr = np.append(arr, state.array_pilas_necesarias())
            continue
    arr = np.append(arr, last_move)
    arr = np.expand_dims(arr, 0)
    return arr


# ESTE METODO RESUELVE UN ESTADO 'state' de 'columns' COLUMNAS EN AL MENOS 'free_moves' PASOS
# UTILIZA LA RED 'clf' PARA PREDECIR MOVIMIENTOS
# 'free_moves' REPRESENTA LA CANTIDAD DE PASOS EN QUE SE LLEGO A LA SOLUCION OPTIMA
# 'n' ES EL MULTIPLICADOR DE 'free_moves' QUE REPRESENTA LA CANTIDAD DE MOVIMIENTOS MAXIMA PARA PODER RESOLVER EL ESTADO
# EL PARAMETRO 'opt' ES LA OPCION PARA SABER QUE ATRIBUTOS SE DESEAN AGREGAR
def nn_solver(free_moves, state, clf, columns, opt):
    movements = generate_movements_array(int(columns))
    n = 3
    # SE TRANSFORMA EL ESTADO
    arr_state = transform_state(state, opt, 0)
    # CUENTA LA CANTIDAD DE MOVIMIENTOS QUE SE HICIERON PARA RESOLVER EL PROBLEMA
    count = 0
    # SE INTENTA RESOLVER EL ESTADO EN 3*FREE_MOVES
    for i in range(n*free_moves):
        j = len(movements) - 1
        count = count + 1
        # GENERO UN PASO CON LA RED
        predict = clf.predict(arr_state)
        order_predict = np.sort(predict[0])
        step = index_mayor(order_predict[j], predict[0])
        lastmove = step
        # SE HACE EL MOVIMIENTO PRODUCIDO POR LA RED
        # EN CASO DE QUE ESTE NO SEA VALIDO, GENERO EL SIGUIENTE PASO QUE PREDIJO LA RED
        while not movement(step, state, movements) or opuesto(movements, lastmove) == step:
            j = j - 1
            step = index_mayor(order_predict[j], predict[0])
        lastmove = step
        arr_state = transform_state(state, "1111", lastmove)
        if state.eval_state():
            return 1, count
    return 0, 0


# 'n' REPRESENTA LA CANTIDAD DE ESTADOS PARA PROBAR A LA RED
# 'opt' REPRESENTA EL STRING "0000" DONDE SE RELLENA CON 1 LOS ATRIBUTOS DESEADOS. ESTOS FUERON DESCRITOS EN EL ARCHIVO
# "data.py"
def nn_test(n, opt):
    columns, rows, ds = dt.get_state_dims("data/states/test/test_f.txt")
    input_dim, output_dim = dt.get_nn_dims(columns, rows, opt)
    clf = nn.load_network(input_dim, output_dim, opt)
    total, success = 0, 0
    for x in range(n):
        total = total + 1
        s = cpmp.init(columns, rows, ds)
        depth = best_solution_depth(s)
        y = nn_solver(depth, s, clf, columns, opt)
        if y[0] == 1:
            print("success")
            success = success + 1
    return "success: "+str(success), "total: "+str(total)







