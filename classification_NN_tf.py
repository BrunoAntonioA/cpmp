from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import os

import tf as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import data as dt

from tensorflow import keras

import numpy as np


train_data, train_labels = dt.import_train_data()
test_data, test_labels = dt.import_test_data()


train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)


clf = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = "data/NN_weights/NN_normalize_gv/nn_16.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


print("train_data.shape: ", train_data.shape)
print("train_data[1].shape: ", train_data[1].shape)
print("train_labels.shape: ", train_labels.shape)
print("test_data.shape: ", test_data.shape)
print("test_labels.shape: ", test_labels.shape)


# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

clf.fit(train_data, train_labels, epochs=150, callbacks=[cp_callback, tensorboard_callback], validation_data=(test_data, test_labels))

print("test data")

test_loss, test_acc = clf.evaluate(test_data,  test_labels, verbose=2)

# Save the weights using the `checkpoint_path` format
clf.save_weights(checkpoint_path.format(epoch=0))

print('\nTest accuracy1:', test_acc)

test_loss, test_acc = clf.evaluate(test_data,  test_labels, verbose=2)

print('\nTest accuracy2:', test_acc)






