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
import xml.etree.ElementTree as ET
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from botocore.client import Config

BUCKET_NAME = '2019-food-recognition-cnn'
FILE_NAME = 'metadata.csv'

# S3 Connect
s3 = boto3.resource('s3',)


def download_all_files(data_dir):
    # select bucket
    bucket = s3.Bucket(BUCKET_NAME)

    if os.path.exists(data_dir): 
        shutil.rmtree(data_dir) 

    os.makedirs(data_dir)
    # download file into current directory
    for s3_object in bucket.objects.all():
        # Need to split s3_object.key into path and file name, else it will give error file not found.
        path, filename = os.path.split(s3_object.key)
        bucket.download_file(s3_object.key, data_dir + '/' + filename)
    


def process_xml(data_dir):
    # Finds all XML files on data/ and append to list
    pascal_voc_contents = []
    owd = os.getcwd()

    os.chdir(data_dir)

    for file in glob.glob("*.xml"):
        f_handle = open(file, 'r')
        pascal_voc_contents.append(xmltodict.parse(f_handle.read()))
    
    list_path_name = []
    list_label=[]

    # Process each file individually
    for index in pascal_voc_contents:
        image_file = index['annotation']['filename']
        # If there's a corresponding file in the folder,
        # process the images and save to output folder
        if os.path.isfile(image_file):
            extract_dataset(index['annotation'],list_path_name, list_label )
        else:
            print("Image file '{}' not found, skipping file...".format(image_file))
    
    os.chdir(owd)
    return list_path_name, list_label

def extract_dataset(dataset, list_path_name, list_label):

    # Open image and get ready to process
    img = Image.open(dataset['filename'])

    # Create output directory
    save_dir = dataset['filename'].split('.')[0]
    try:
        os.mkdir(save_dir)
    except:
        pass
    # Image name preamble
    sample_preamble = save_dir + "/" + dataset['filename'].split('.')[0] + "_"
    # Image counter
    i = 0

    # Run through each item and save cut image to output folder

    if (type(dataset['object']) is list):

        for item in dataset['object']:    
            #Save label
            label = item['name'] 
            list_label.append(label)

            # Convert str to integers
            bndbox = dict([(a, int(b)) for (a, b) in item['bndbox'].items()])
            # Crop image
            im = img.crop((bndbox['xmin'], bndbox['ymin'],
                        bndbox['xmax'], bndbox['ymax']))
            # Save
            nameImagen =sample_preamble + str(i) + '.jpg' 
            im.save(nameImagen)
            i = i + 1

            #Save path name
            list_path_name.append(nameImagen)
    else:
        list_label.append(dataset['object']['name'])

        # Convert str to integers
        bndbox = dict([(a, int(b)) for (a, b) in dataset['object']['bndbox'].items()])
        # Crop image
        im = img.crop((bndbox['xmin'], bndbox['ymin'],
                       bndbox['xmax'], bndbox['ymax']))
        # Save
        nameImagen =sample_preamble + str(i) + '.jpg' 
        im.save(nameImagen)
        i = i + 1

        #Save path name
        list_path_name.append(nameImagen)
        
    return list_path_name, list_label
        
  

def prepare_dataset(data_dir):
    pathName, labels = process_xml(data_dir)

    fn = lambda x: os.path.basename(x)
    names = [fn(name) for name in pathName]

    splits = ['train' if random.random() <= 0.7 else 'valid' for _ in names]

    #Create Json
    data = {}
    index = 0
    for label in labels:
        if(label not in data.keys()):
            data[label] = index 
            index += 1

    #Save Json
    with open(os.path.join('labelMap.json'), 'w') as outfile:
        json.dump(data, outfile)

    labels_ids = [data[x] for x in labels ] 
    metadata = pd.DataFrame({'label': labels, 'image_name': names, 'split': splits,'path':pathName, 'label_id': labels_ids})
    metadata.to_csv(os.path.join(data_dir, FILE_NAME), index=False)


def main (dir_path):
    
    print('Download files ... ')
    download_all_files(dir_path)
    print('Done')
    
    print('Prepare dataSet ... ')
    prepare_dataset(dir_path)
    print('Done')

    print_dataset(dir_path)


def print_dataset(dir_path):
    metadata = pd.read_csv(os.path.join(dir_path, FILE_NAME))
    exclude_labels =["bread", "crepe", "lettuce", "soup", "french frie", "fish", "sauce", "lemon", "tomato", "bean", "broccoli", "carrot", "sushi", "coffee", "potato", "fried plantain", "ripe banana", "biscuit", "meat pie", "sausage", "cheese", "pasta", "sandwich", "onion", "hamburger", "jelly", "cake", "pineapple", "ham", "pizza", "tree tomato", "pork", "grape", "pancakes", "cape gooseberry", "dragon fruit", "peach", "chocolate", "guava", "bacon", "passionflower", "ice cream", "banana", "passion fruit"] 
    train_sources = build_sources_from_metadata(metadata, dir_path, exclude_labels=exclude_labels)
    valid_sources = build_sources_from_metadata(metadata, dir_path, mode='valid', exclude_labels=exclude_labels)

    print(train_sources[:10])
 
    dataset = make_dataset(train_sources, training=True,
    batch_size=3, num_epochs=1,
    num_parallel_calls=3)
    dataset = iter(dataset)

    #imshow_batch_of_three(next(dataset))

def preprocess_image(image):
    image = tf.image.resize(image, size=(200, 200))
    image = image / 255.0
    return image

def augment_image(image):
    return image

def make_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
    """
    Returns an operation to iterate over the dataset specified in sources

    Args:
        sources (list): A list of (filepath, label_id) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.

    Returns:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """

    def load(row):
        filepath = row['image']
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img)
        return img, row['label']

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4

    images, labels = zip(*sources)

    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'label': list(labels)}) 

    if training:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(lambda x,y: (preprocess_image(x), y))

    if training:
        ds = ds.map(lambda x,y: (augment_image(x), y))
        
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds

def build_sources_from_metadata(metadata, data_dir, mode='train', exclude_labels=None):
    
    if exclude_labels is None:
        exclude_labels = set()
    if isinstance(exclude_labels, (list, tuple)):
        exclude_labels = set(exclude_labels)

    df = metadata.copy()
    df = df[df['split'] == mode]
    df['filepath'] = df['path'].apply(lambda x: os.path.join(data_dir, x))
    include_mask = df['label_id'].apply(lambda x: x not in exclude_labels)

    df = df[include_mask]

    sources = list(zip(df['filepath'], df['label_id']))
    return sources

def imshow_batch_of_three(batch, show_label=True):

    with open('labelMap.json', 'r') as read_file:
        data = json.load(read_file)

    label_batch = [get_label_by_id(data,label_id) for label_id in batch[1].numpy()] 
    image_batch = batch[0].numpy()
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i in range(3):
        img = image_batch[i, ...]
        
        axarr[i].imshow(img)
        if show_label:
            axarr[i].set(xlabel='label = {}'.format(label_batch[i]))
    #plt.show()     


def get_label_by_id(json_object, label_id):
    for label in json_object:
        if json_object[label] == label_id:
            return label
