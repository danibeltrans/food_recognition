import os
import boto3
import sys
import random
import shutil
import json
import xmltodict
import glob

import botocore
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import data as dt

BATCH_SIZE = 4
VALIDATION_STEPS = 2
NUMBER_CLASSES=17

def get_label():
    labels = []
    with open('labelMap.json','r') as read_file:
        data = json.load(read_file)

    for label in data:
        labels.append(label)

    return label

def imshow_with_predictions(model, batch, show_label=True):

    with open('labelMap.json','r') as read_file:
        data = json.load(read_file)

    label_batch = [dt.get_label_by_id(data,label_id) for label_id in batch[1].numpy()] 
    image_batch = batch[0].numpy()
    pred_batch = model.predict(image_batch)
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        pred = int(np.argmax(pred_batch[i]))
        pred = dt.get_label_by_id(data, pred)
        msg = f'pred = {pred}'
        if show_label:
            msg += f', label = {label_batch[i]}'
        axarr[i].set(xlabel=msg)

    #plt.show()

def linear_model(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(200,200,3)))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def build_standard_cnn(
    config_architecture=[0, 16, 0, 120],
    num_units_per_dense = 1,
    filter=6,
    kernel_size=(5,5), 
    activation='tanh',
    input_shape=(200, 200, 3),
    num_classes=17,
    num_units_per_dense_layer =[84, 80],
    pooling_method= 'AveragePooling' ):

    """
    Returns a convolutional model with the configuration selected   

    Args:
        config_architecture (list): Architecture
        num_units_per_dense (int): Number per dense
        filter (int): The dimensionality of the output space
        kernel_size (tuple): Kernel size 
        activation (String): Activation funtion (tanh, relu)
        input_shape (tuple): Input shape
        num_classes (int): Number of classes

    """

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=filter, kernel_size=kernel_size, activation=activation,
            padding='same', input_shape=input_shape
        )
    )
    for num_filter in config_architecture:
        if (num_filter == 0):
            if (pooling_method == 'AveragePooling'):
                model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

            if (pooling_method == 'MaxPooling'):
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        else:
            model.add(tf.keras.layers.Conv2D(num_filter, (3,3), activation=activation, padding='same'))
 
    model.add(tf.keras.layers.Flatten())

    for num_units in num_units_per_dense_layer:
        model.add(tf.keras.layers.Dense(num_units, activation=activation))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model


def lenet5 (num_classes):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=6, kernel_size=(5,5), activation='tanh',
            padding='same', input_shape=(200, 200, 3)
        )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation='tanh'))
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(120, (5,5), activation='tanh'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model

def alex_net (num_classes):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=98, kernel_size=(11,11), activation='relu',
            padding='same', input_shape=(200, 200, 3)
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(256, (5,5), activation='relu',padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(384, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(384, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model

def vgg_net (num_classes):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same', input_shape=(200, 200, 3)
        )
    )
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu',padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model

def main(dir_path, model_name=None):
    metadata = pd.read_csv(os.path.join(dir_path,'metadata.csv'))
    exclude_labels =["bread", "crepe", "lettuce", "soup", "french frie", "fish", "sauce", "lemon", "tomato", "bean", "broccoli", "carrot", "sushi", "coffee", "potato", "fried plantain", "ripe banana", "biscuit", "meat pie", "sausage", "cheese", "pasta", "sandwich", "onion", "hamburger", "jelly", "cake", "pineapple", "ham", "pizza", "tree tomato", "pork", "grape", "pancakes", "cape gooseberry", "dragon fruit", "peach", "chocolate", "guava", "bacon", "passionflower", "ice cream", "banana", "passion fruit"] 
    train_sources = dt.build_sources_from_metadata(metadata, dir_path, exclude_labels=exclude_labels)
    valid_sources = dt.build_sources_from_metadata(metadata, dir_path, mode='valid', exclude_labels=exclude_labels)
 
    model = 0
    if model_name == 'linear':
        model = linear_model(NUMBER_CLASSES)
    elif model_name == 'lenet':
        model = lenet5(NUMBER_CLASSES)
    elif model_name == 'alexNet':
        model = alex_net(NUMBER_CLASSES)
    elif model_name == 'vggNet':
        model = vgg_net(NUMBER_CLASSES)
    else:  
        print('Buid Standard Cnn') 
        model = build_standard_cnn()

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.optimizers.Adam(0.0001),
                metrics=['accuracy'])
    model.summary()

    train_dataset = dt.make_dataset(train_sources, training=True,
        batch_size=BATCH_SIZE, num_epochs=8,
        num_parallel_calls=2)
    valid_dataset = dt.make_dataset(valid_sources, training=False,
        batch_size=BATCH_SIZE, num_epochs=1,
        num_parallel_calls=2)   

    training(model, train_dataset, valid_dataset)

    dataset = dt.make_dataset(valid_sources, training=False,
    batch_size=3, num_epochs=10,
    num_parallel_calls=2)
    dataset = iter(dataset)
    #imshow_with_predictions(model, next(dataset))

def training(model, train_dataset, valid_dataset):
    
    history = model.fit(x=train_dataset, epochs=10,
        validation_data=train_dataset, validation_steps=VALIDATION_STEPS)

    print("\n")
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # print('----------REPORT ON TRAINING DATA-----------')
    # print(model.evaluate(train_dataset))
    # print('----------REPORT ON VALIDATION DATA-----------')
    # print(model.evaluate(valid_dataset))
