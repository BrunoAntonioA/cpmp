import numpy as np


def import_train_data():
    train = open("states/test.txt", "r")
    labels = open("states/testlabel.txt", "r")
    train_data = train.read()
    labels_data = labels.read()
    inputs = []
    outputs = []
    state = np.array([])
    aux = np.array([])
    for i in train_data:
        if i != '-' and i != ',':
            aux = np.append(aux, int(i))
            continue
        if i == '-':
            helper = 0
            for x in aux:
                helper = helper * 10 + x
            state = np.append(state, helper)
            aux = np.array([])
            continue
        if i == ',':
            inputs.append(state.astype(int))
            print("state: ", state)
            state = np.array([])
            continue
        state = np.append(state, int(i))

    for i in labels_data:
        if i == ',':
            continue
        outputs.append(int(i))
    train.close()
    labels.close()
    return inputs, outputs


def import_test_data():
    train = open("data/states/test.txt", "r")
    labels = open("data/states/testlabel.txt", "r")
    train_data = train.read()
    labels_data = labels.read()
    inputs = []
    outputs = []
    state = []
    for i in train_data:

        if i == ',':
            inputs.append(np.asarray(state))
            state = []
            continue
        state.append(int(i))

    for i in labels_data:
        if i == ',':
            continue
        outputs.append(int(i))
    train.close()
    labels.close()
    return inputs, outputs

