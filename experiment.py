import json
import model as md
import data as data

#load variables from json config

with open('config.json', 'r') as file:
    config = json.load(file)

model_name = config['DEFAULT']['model_name']
num_classes = config['DEFAULT']['num_classes']
DATA_DIR = config['DEFAULT']['DATA_DIR']


#load metadata, load dataset

data.main (DATA_DIR)

#load model, fit model

md.main(DATA_DIR, model_name, num_classes)

#save learning curves

