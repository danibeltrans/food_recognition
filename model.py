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

BATCH_SIZE = 256
VALIDATION_STEPS = 2


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

    plt.show()

def linear_model(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(200,200,3)))
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

def main(dir_path, model_name = 'alexNet'):
    metadata = pd.read_csv(os.path.join(dir_path,'metadata.csv'))

    train_sources = dt.build_sources_from_metadata(metadata, dir_path)
    valid_sources = dt.build_sources_from_metadata(metadata, dir_path, mode='valid')
 
    model = 0
    if model_name == 'linear':
        model = linear_model(16)
    elif model_name == 'lenet':
        model = lenet5(16)
    elif model_name == 'alexNet':
        model = alex_net(16)
    elif model_name == 'vggNet':
        model = vgg_net(16)
    else:   
        model=alex_net(16)

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.optimizers.Adam(0.0001),
                metrics=['accuracy'])
    model.summary()

    train_dataset = dt.make_dataset(train_sources, training=True,
        batch_size=BATCH_SIZE, num_epochs=10,
        num_parallel_calls=2)
    valid_dataset = dt.make_dataset(valid_sources, training=False,
        batch_size=BATCH_SIZE, num_epochs=1,
        num_parallel_calls=2)   

    training(model, train_dataset, valid_dataset)

    dataset = dt.make_dataset(valid_sources, training=False,
    batch_size=3, num_epochs=1,
    num_parallel_calls=2)
    dataset = iter(dataset)
    imshow_with_predictions(model, next(dataset))

def training(model, train_dataset, valid_dataset):
    
    model.fit(x=train_dataset, epochs=10,
        validation_data=valid_dataset, validation_steps=VALIDATION_STEPS)
        #validation_data=train_dataset, validation_steps=VALIDATION_STEPS)

    print('----------REPORT ON TRAINING DATA-----------')
    print(model.evaluate(train_dataset))
    print('----------REPORT ON VALIDATION DATA-----------')
    print(model.evaluate(valid_dataset))

