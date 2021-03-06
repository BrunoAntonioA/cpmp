from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import data as dt

from tensorflow import keras

import numpy as np

"""
config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
"""
path1 = "data/states/test.txt";
path2 = "data/states/testlabel.txt";
path3 = "data/states/train.txt";
path4 = "data/states/trainlabel.txt";

train_data, train_labels = dt.import_data(path3, path4)
test_data, test_labels = dt.import_data(path1, path2)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

opt = "0010"
train_data, input_dim = dt.filter_data(train_data, 4, 4, opt)
test_data, input_dim = dt.filter_data(test_data, 4, 4, opt)

print("input_dim: ", input_dim)

clf = keras.Sequential([
    keras.layers.Flatten(input_shape=(input_dim,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


checkpoint_path = "data/NN_weights/NN_normalize_gv/nn_16.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

print("compiled succesfuly")


# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

clf.fit(train_data, train_labels, epochs=150, callbacks=[cp_callback, tensorboard_callback],
        validation_data=(test_data, test_labels))

print("test data")

test_loss, test_acc = clf.evaluate(test_data,  test_labels, verbose=2)

# Save the weights using the `checkpoint_path` format
clf.save_weights(checkpoint_path.format(epoch=0))

print('\nTest accuracy1:', test_acc)

test_loss, test_acc = clf.evaluate(test_data,  test_labels, verbose=2)

print('\nTest accuracy2:', test_acc)






