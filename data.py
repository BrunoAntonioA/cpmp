import numpy as np


def import_data(path, path2):
    train = open(path, "r")
    labels = open(path2, "r")
    train_data = train.read()
    labels_data = labels.read()
    inputs = []
    outputs = []
    pos = ""
    sta = np.array([])
    for i in train_data:
        if i != '.' and i != ',':
            pos = pos + str(i)
            continue
        if i == '.':
            sta = np.append(sta, pos)
            pos = ""
            continue
        if i == ',':
            sta = np.append(sta, pos)
            pos = ""
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
    return inputs, outputs


def filter_data(data, c, opt):
    data = np.array(data)
    print("data.shape: ", data.shape)

    for x in range(len(opt)):
        if x == 0 and opt[x] == "1":
            print("group value")
        if x == 1 and opt[x] == "1":
            print("base differences")
        if x == 2 and opt[x] == "1":
            print("top differences")
        if x == 3 and opt[x] == "1":
            print("necessaries stacks")

path = "data/states/test.txt";
path2 = "data/states/testlabel.txt";

inputs, outputs = import_data(path, path2)

filter_data(inputs, 5, "0000")

