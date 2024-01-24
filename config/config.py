print('config.py')
"""
# Creation date 301122t11h44
# config.py
# Configuration file for the application
# 
"""
merge_features = {
    'heart_related': [
        'Heart Rate', 
        'Mean blood pressure', 
        'Blood pressure systolic', 
        'Blood pressure diastolic'
    ],
    'comma_related': [
        'Glascow coma scale eye opening',
        'Glascow coma scale verbal response',
        'Glascow coma scale motor response',
        'Glascow coma scale total'
    ],
    'respiratory_related': [
        'Respiratory rate',
        'Oxygen saturation'
    ],
    'pH': [],
    'temperature': [],
    'BMI': [],
    'Capillary refill rate': [],
    'Glucose': [],
}    

orig_rules = {
        'Oxygen saturation': [
            lambda x : x < 95, 
            lambda x : x >= 95],
        'Heart Rate': [
            lambda x : x < 60, 
            lambda x : (x>=60 and x <= 100),  
            lambda x : x > 100 
        ],
        'BMI': [
            lambda x : x < 16.5, 
            lambda x : (x >= 16.5 and x < 25), 
            lambda x : (x >= 25 and x < 30), 
            lambda x : (x >= 30 and x < 35), 
            lambda x : (x >= 35 and x < 40),
            lambda x : x >= 40
        ],
        'Glucose': [
            lambda x : x < 72, 
            lambda x : (x >= 72 and x <= 108), 
            lambda x : (x > 108 and x <= 125), 
            lambda x : x > 125, 
        ],
        'Capillary refill rate': [
            lambda x: x < 3,
            lambda x : (x >= 3), 
        ],
        'Systolic blood pressure': [
            lambda x: x < 90,
            lambda x : (x >= 90 and x <= 120),
            lambda x : (x >= 120 and x <= 129),
            lambda x : (x >= 130 and x <= 139),
            lambda x : x > 139,
        ],
        'Diastolic blood pressure': [
            lambda x : x < 60,
            lambda x : (x >= 60 and x <= 80),
            lambda x : (x > 80 and x <= 89),
            lambda x : (x > 89),
        ],
        'Mean blood pressure': [
            lambda x : x < 60,
            lambda x : (x >= 60 and x < 110),
            lambda x : (x >= 110 and x < 160),
            lambda x : (x >= 160),
            
        ],
        'Glascow coma scale eye opening': [
            lambda x: x < 4,
            lambda x: x >= 4,
        ],
        'Glascow coma scale verbal response': [
            lambda x: x < 5,
            lambda x: x >= 5,
        ],
        'Glascow coma scale motor response': [
            lambda x: x < 6,
            lambda x: x >= 6,    
        ],
        'Glascow coma scale total': [
            lambda x: x < 15,
            lambda x: x >= 15
        ],
        'Respiratory rate': [
            lambda x: x < 12,
            lambda x: (x >= 12 and x <= 20),
            lambda x: x > 20
        ],
        'Temperature': [
            lambda x: x < 35,
            lambda x: (x >= 35 and x <= 36.5),
            lambda x: (x > 36.5 and x <= 37.5),
            lambda x: (x > 37.5 and x <= 38.3),
            lambda x: (x > 38.3 and x <= 40),
            lambda x: (x > 40),
        ],
        'pH': [
            lambda x: x < 7.35,
            lambda x: (x >= 7.35 and x <= 7.45),
            lambda x: x > 7.45        
        ]

    }

rules = {
        'Oxygen saturation': [
            lambda x : x < 95, 
            lambda x : x >= 95],
        'Heart Rate': [
            lambda x : x < 60, 
            lambda x : (x>=60 and x <= 100),  
            lambda x : x > 100 
        ],
        'BMI': [
            lambda x : x < 18.5, 
            lambda x : (x >= 18.5 and x < 25), 
            lambda x : x >= 25
        ],
        'Glucose': [
            lambda x : x < 72, 
            lambda x : (x >= 72 and x <= 108), 
            lambda x : x > 108, 
        ],
        'Capillary refill rate': [
            lambda x: x < 3,
            lambda x : (x >= 3), 
        ],
        'Systolic blood pressure': [
            lambda x: x < 90,
            lambda x : (x >= 90 and x <= 120),
            lambda x : x > 120,
        ],
        'Diastolic blood pressure': [
            lambda x : x < 60,
            lambda x : (x >= 60 and x <= 80),
            lambda x : (x > 80),
        ],
        'Mean blood pressure': [
            lambda x : x < 60,
            lambda x : (x >= 60 and x < 110),
            lambda x : (x >= 110),
            
        ],
        'Glascow coma scale eye opening': [
            lambda x: x < 4,
            lambda x: x >= 4,
        ],
        'Glascow coma scale verbal response': [
            lambda x: x < 5,
            lambda x: x >= 5,
        ],
        'Glascow coma scale motor response': [
            lambda x: x < 6,
            lambda x: x >= 6,    
        ],
        'Glascow coma scale total': [
            lambda x: x < 15,
            lambda x: x >= 15
        ],
        'Respiratory rate': [
            lambda x: x < 12,
            lambda x: (x >= 12 and x <= 20),
            lambda x: x > 20
        ],
        'Temperature': [
            lambda x: x < 35,
            lambda x: (x >= 35 and x <= 36.5),
            lambda x: (x > 36.5),
        ],
        'pH': [
            lambda x: x < 7.35,
            lambda x: (x >= 7.35 and x <= 7.45),
            lambda x: x > 7.45        
        ]

    }

rules_normal = {
    'Oxygen saturation': 1,
    'Heart Rate': 1,
    'BMI': 1,
    'Glucose': 1,
    'Capillary refill rate': 0,
    'Systolic blood pressure': 1,
    'Diastolic blood pressure': 1,
    'Mean blood pressure': 1,
    'Glascow coma scale eye opening': 1,
    'Glascow coma scale verbal response': 1,
    'Glascow coma scale motor response': 1,
    'Glascow coma scale total': 1,
    'Respiratory rate': 1,
    'Temperature': 1,
    'pH': 1
}

all_mappings = {
    'Glascow coma scale verbal response': {
    # Glascow coma scale verbal response
    '1 No Response': ['No Response', 'No Response-ETT', '1.0 ET/Trach'],
    '2 Incomp sounds': ['Incomprehensible sounds'],
    '3 Inapprop words': ['Inappropriate Words'],
    '4 Confused': ['Confused'], 
    '5 Oriented': ['Oriented'],
    },
    'Glascow coma scale eye opening' : {
        '1 No Response': ['None'],
        '2 To pain': ['To Pain'],
        '3 To speech': ['To Speech'],
        '4 Spontaneously': ['Spontaneously'],
    },

    'Glascow coma scale motor response': {
        '1 No Response': ['No response'],
        '2 Abnorm extensn': ['Abnormal extension'],
        '3 Abnorm flexion': ['Abnormal Flexion'],
        '4 Flex-withdraws': ['Flex-withdraws'],
        '5 Localizes Pain': ['Localizes Pain'],
        '6 Obeys Commands': ['Obeys Commands']
    }
}


##

'''
070722 21h42
4)  load dat

res = get_data.load_save_pickle_files_parallel(['train', 'test'])
train = res[0]
test = res[1]

070722 20h46 parallel loading test get_data
Return data train_df, val_df, test_df
'''
# s220622.py
# "https://github.com/pyg-team/pytorch_geometric"
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import itertools
import matplotlib.pyplot as plt
import matplotlib as mplt # colors,patches
import pickle
import os
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
import time
import argparse
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch_geometric

# from my

files_in_dir = os.listdir(os.getcwd())
dbg_cwfname = 'get_data.py'
# print('Using DATA_DIR: {}'.format(DATA_DIR))
DATA_DIR="./data"
dataset_names = os.listdir(DATA_DIR)
WORKING_DATASET_PATH = os.path.join(DATA_DIR, 'phenotyping')
TRAIN_DATASET_PATH = os.path.join(WORKING_DATASET_PATH, 'train')
TEST_DATASET_PATH = os.path.join(WORKING_DATASET_PATH, 'test')
WORKING_DATASET_PATH_TRAIN = os.path.join(WORKING_DATASET_PATH, 'train')
WORKING_DATASET_PATH_TEST = os.path.join(WORKING_DATASET_PATH, 'test')

config = {
    "data_dir": DATA_DIR,
    "working_dataset_path": WORKING_DATASET_PATH,
    "train_dataset_path": TRAIN_DATASET_PATH,
    "test_dataset_path": TEST_DATASET_PATH,
    "working_dataset_path_train": WORKING_DATASET_PATH_TRAIN,
    "working_dataset_path_test": WORKING_DATASET_PATH_TEST
}
    

import argparse 
args = argparse.Namespace()
argparser = argparse.ArgumentParser()

argparser.add_argument('--data_dir', type=str, default=DATA_DIR)
argparser.add_argument('--working_dataset_path', type=str, default=WORKING_DATASET_PATH)
argparser.add_argument('--train_dataset_path', type=str, default=TRAIN_DATASET_PATH)
argparser.add_argument('--test_dataset_path', type=str, default=TEST_DATASET_PATH)
argparser.add_argument('--working_dataset_path_train', type=str, default=WORKING_DATASET_PATH_TRAIN)
argparser.add_argument('--working_dataset_path_test', type=str, default=WORKING_DATASET_PATH_TEST)
args = argparser.parse_args()

print('after virus check')

random_number = 121