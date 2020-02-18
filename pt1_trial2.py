#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:07:19 2020

@author: sarahpell
"""


import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
# from keras_drop_block import DropBlock2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import optimizers

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.client import device_lib
import pickle

def plot_acc_loss(history):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    plt.savefig('pt1_alexnet_plt.png')

def build_neural_network():

    np.random.seed(1000)
    #Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding="valid"))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation("relu"))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation("relu"))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation("relu"))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(250))
    model.add(Activation("softmax"))

    return model

def available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


# def main(batch_size=128,epochs=1,val_frac=0.2,):
batch_size=256
epochs=100
val_frac=0.2

data_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25)

train_it = data_gen.flow_from_directory("png/", target_size=(224, 224), class_mode = "categorical", batch_size = 15000, subset = "training")
test_it = data_gen.flow_from_directory("png/", target_size = (224,224), class_mode = "categorical", batch_size = 5000, subset = "validation")

x_train, y_train = train_it.next()
x_test, y_test = test_it.next()

inds = np.random.permutation(len(x_train))
split_ind = int(len(x_train) * (1 - val_frac))
train_inds, val_inds = inds[:split_ind], inds[split_ind:]
x_val, y_val = x_train[val_inds], y_train[val_inds]
x_train, y_train = x_train[train_inds], y_train[train_inds]

print(f"Train shape: {x_train.shape}")
print(f"Validation shape: {x_val.shape}")
print(f"Test shape: {x_test.shape}")



model = build_neural_network()
sgd = optimizers.SGD(lr=0.01)
model.compile(
  optimizer=sgd,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)
model.summary()


# Distribute the neural network over multiple GPUs if available.
gpu_count = len(available_gpus())
if gpu_count > 1:
    print(f"\n\nModel parallelized over {gpu_count} GPUs.\n\n")
    parallel_model = keras.utils.multi_gpu_model(model, gpus=gpu_count)
else:
    print("\n\nModel not parallelized over GPUs.\n\n")
    parallel_model = model

parallel_model.compile(
  optimizer=sgd,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

checkpoint = keras.callbacks.ModelCheckpoint(
  "pt1_weights2.h5",
  monitor="val_acc",
  save_weights_only=True,
  save_best_only=True,
)


history = parallel_model.fit(
  x_train,
  y_train,
  batch_size=batch_size,
  epochs=epochs,
  verbose=1,
  validation_data=(x_val, y_val),
  callbacks=[checkpoint],
)

with open('pt1_history.pickle', 'wb') as handle:
    pickle.dump(history, handle)

# plot_acc_loss(history)


parallel_model.load_weights("pt1_weights2.h5")
score = parallel_model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
print(f"Test score:    {score[0]: .4f}")
print(f"Test accuracy: {score[1] * 100.:.2f}")


preds = parallel_model.predict(x_test, batch_size=batch_size)



parallel_model.save('alexnet_pt1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

# plt.figure()
plt.savefig('pt1_accplot_optim.png')
plt.clf()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# plt.show()
plt.savefig('pt1_lossplot_optim.png')

