import numpy as np


# SE IMPORTAN LOS DATOS DEL ARCHIVO DE TEXTO GENERADO POR EL METODO "def generate_data_set(n1, n2, c, f, ds)" QUE SE
# ENCUENTRA EN EL ARCHIVO "CPMP.py"
def import_data(path, path2):
    train = open(path, "r")
    labels = open(path2, "r")
    train_data = train.read()
    labels_data = labels.read()
    inputs = []
    outputs = []
    pos = 0
    sta = np.array([], dtype="int")
    c, f, ds = 0, 0, 0
    cf_count = 0
    for i in train_data:
        if cf_count < 3:
            if i != '/':
                if cf_count == 0:
                    c = c*10 + int(i)
                    continue
                if cf_count == 1:
                    f = f*10 + int(i)
                    continue
                if cf_count == 2:
                    ds = ds*10 + int(i)
                    continue
            else:
                cf_count = cf_count + 1
                continue
        if i != '.' and i != ',':
            pos = pos * 10 + int(i)
            continue
        if i == '.':
            sta = np.append(sta, int(pos))
            pos = 0
            continue
        if i == ',':
            sta = np.append(sta, int(pos))
            pos = 0
            inputs.append(sta)
            sta = []
            continue
    pos = ""
    for i in labels_data:
        if i == ',':
            outputs.append(pos)
            pos = ""
            continue
        pos = pos + i
    train.close()
    labels.close()
    return np.array(inputs), outputs, c, f


# METODO PARA FILTRAR DATOS CARGADOS CON EL FORMATO DE IMPORT_DATA
# EL PARAMETRO 'opt' ES EL ENCARGADO DE DECIR QUE PARAMETROS SON LOS QUE LLEVARA EL ESTADO
# EL FORMATO DE LA VARIABLE 'opt' ES DE UN STRING CON 4 DIGITOS "0000" EN UN ESPECIFICO ORDEN SI ES QUE EL DIGITO SE
# ENCUENTRA CON EL NUMERO '1' INDICARA LA PRESENCIA DE ESTE ATRIBUTO
# EL ORDEN ES EL SIGUIENTE: group_value_array, base_differences, top_differences y necessaries_stacks
"""
    Los atributos generados para cada estado son:
        group_value_array: Indica la cantidad de contenedores desordenados de cada columna.
        get_base_differences: Indica la diferencia del contenedor de la base con el mayor de todos los container.
        get_top_differences: Indica la diferencia del contenedor de la base con el mayor de todos los container.
        necessaries_stacks: Indica la cantidad de pilas necesarias para
"""


def filter_data(data, c, f, opt):
    data = np.array(data)
    transformed_data = []

    for dt in data:
        new_data = np.array([])
        state = np.split(dt, [0, c*f])
        state = state[1]
        new_data = np.append(new_data, state)
        output_cont = 0
        for x in range(len(opt)):
            if x == 0 and opt[x] == "1":
                initial_split_position = c * f
                final_split_position = (c*f) + c
                group_value = np.split(dt, [initial_split_position, final_split_position])
                group_value = group_value[1]
                new_data = np.append(new_data, group_value)
                output_cont = output_cont + 1
            if x == 1 and opt[x] == "1":
                initial_split_position = (c*f) + c
                final_split_position = (c * f) + (2 * c)
                base_differences = np.split(dt, [initial_split_position, final_split_position])
                base_differences = base_differences[1]
                new_data = np.append(new_data, base_differences)
                output_cont = output_cont + 1
            if x == 2 and opt[x] == "1":
                initial_split_position = (c * f) + (2 * c)
                final_split_position = (c * f) + (3 * c)
                top_differences = np.split(dt, [initial_split_position, final_split_position])
                top_differences = top_differences[1]
                new_data = np.append(new_data, top_differences)
                output_cont = output_cont + 1
            if x == 3 and opt[x] == "1":
                initial_split_position = (c * f) + (3 * c)
                final_split_position = (c * f) + (4 * c)
                necessaries_stacks = np.split(dt, [initial_split_position, final_split_position])
                necessaries_stacks = necessaries_stacks[1]
                new_data = np.append(new_data, necessaries_stacks)
                output_cont = output_cont + 1
        last_move = dt[len(dt)-1]
        new_data = np.append(new_data, last_move)
        transformed_data.append(new_data)
        output_dim = (c-1)*c

    return np.array(transformed_data), (transformed_data[0].shape[0],), output_dim


# RETORNA LAS DIMENSIONES DE LOS ESTADOS GUARDADOS EN EL ARCHIVO DE TEXTO
def get_state_dims(path):
    train = open(path, "r")
    train_data = train.read()
    cont = 0
    row = 0
    ds = 0
    column = 0
    for x in train_data:
        if cont == 0:
            if x == '/':
                cont = cont + 1
                continue
            else:
                column = column*10 + int(x)
                continue
        if cont == 1:
            if x == '/':
                cont = cont + 1
                continue
            else:
                row = row*10 + int(x)
        if cont == 2:
            if x == '/':
                break
            else:
                ds = ds * 10 + int(x)
    train.close()
    return column, row, ds


# RETORNA LAS DIMENSIONES UTILIZADAS PARA LA RED, EL PARAMETRO 'opt' CUMPLE LA MISMA FUNCION QUE EN EL METODO
# "filter_data()"
def get_nn_dims(columns, rows, opt):
    dim_cont = 0
    for x in range(len(opt)):
        if x == 0 and opt[x] == "1":
            dim_cont = dim_cont + 1
        if x == 1 and opt[x] == "1":
            dim_cont = dim_cont + 1
        if x == 2 and opt[x] == "1":
            dim_cont = dim_cont + 1
        if x == 3 and opt[x] == "1":
            dim_cont = dim_cont + 1
    columns = int(columns)
    rows = int(rows)
    input_dim = (columns * rows) + (dim_cont * columns) + 1
    output_dim = (columns-1)*columns
    return (input_dim,), int(output_dim)


