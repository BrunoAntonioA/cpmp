import numpy as np


def import_data(path, path2):
    train = open(path, "r")
    labels = open(path2, "r")
    train_data = train.read()
    labels_data = labels.read()
    inputs = []
    outputs = []
    pos = 0
    sta = np.array([], dtype="int")
    for i in train_data:
        if i != '.' and i != ',':
            pos = pos * 10 + int(i)
            continue
        if i == '.':
            sta = np.append(sta, int(pos))
            pos = 0
            print("\nsta: ", sta)
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
    return np.array(inputs), outputs


def filter_data(data, c, f, opt):
    data = np.array(data)
    transformed_data = []

    for dt in data:
        new_data = np.array([])
        state = np.split(dt, [0, c*f])
        state = state[1]
        new_data = np.append(new_data, state)
        for x in range(len(opt)):
            if x == 0 and opt[x] == "1":
                initial_split_position = c * f
                final_split_position = (c*f) + c
                group_value = np.split(dt, [initial_split_position, final_split_position])
                group_value = group_value[1]
                new_data = np.append(new_data, group_value)
            if x == 1 and opt[x] == "1":
                initial_split_position = (c*f) + c
                final_split_position = (c * f) + (2 * c)
                base_differences = np.split(dt, [initial_split_position, final_split_position])
                base_differences = base_differences[1]
                new_data = np.append(new_data, base_differences)
            if x == 2 and opt[x] == "1":
                initial_split_position = (c * f) + (2 * c)
                final_split_position = (c * f) + (3 * c)
                top_differences = np.split(dt, [initial_split_position, final_split_position])
                top_differences = top_differences[1]
                new_data = np.append(new_data, top_differences)
            if x == 3 and opt[x] == "1":
                initial_split_position = (c * f) + (3 * c)
                final_split_position = (c * f) + (4 * c)
                necessaries_stacks = np.split(dt, [initial_split_position, final_split_position])
                necessaries_stacks = necessaries_stacks[1]
                new_data = np.append(new_data, necessaries_stacks)
        last_move = dt[len(dt)-1]
        new_data = np.append(new_data, last_move)
        transformed_data.append(new_data)

    return np.array(transformed_data), transformed_data[0].shape[0]



