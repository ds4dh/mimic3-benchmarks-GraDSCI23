
from scipy.sparse import csr_matrix
import numpy as np 
from tqdm import tqdm
import inspect 
import os 
import itertools as it 
import time
import pickle
import math
import pandas as pd 

import sys
from gnn__models.config.config import merge_features, orig_rules, rules, rules_normal, all_mappings 
from gnn__models.connectivity_strategies.common_code_for_strategies import create_category_files

# TODO make configurable 
FOLDER_STRUCTURE = 'data/phenotyping/'
# parse parameters 
import argparse
args = argparse.ArgumentParser()
# small_part default True
args.add_argument('--small_part', type=bool, default=False)
args.add_argument('--clean', type=bool, default=False)
clean = args.parse_args().clean
small_part = args.parse_args().small_part
print('small_part', small_part)
print(f'Phase 1')
spstr = '_small_part' if small_part else ''

if clean: 
    '''execute bash lines 
    rm dict_dfs_cat.pk
    rm alldfs_categories.csv
    rm A_expert_edges_exact.pk
    '''
    os.system(f'rm dict_dfs_cat{spstr}.pk')
    os.system(f'rm alldfs_categories{spstr}.csv')
    os.system(f'rm A_expert_edges_exact{spstr}.pk')

create_category_files(FOLDER_STRUCTURE, spstr)
# fname = f'dict_dfs_cat.pk{spstr}'
# if not os.path.exists(fname):
#     print(f'Creating: fname: {fname}')
#     # 1m 
#     # parse all files in the folder path recursively
#     def parse_folder(path):
#         count = 0
        
#         for root, dirs, files in os.walk(path):
#             for file in files:
#                 count += 1 
#                 if count % 10000 == 0:
#                     print(count)
#                 if count > 60000:
#                     raise Exception('too many something is wrong ')
#                 if 'episode' in file:
#                     yield os.path.join(root, file)

#     filenames = []
#     dict_dfs = {}

#     assert os.path.exists(FOLDER_STRUCTURE), f'folder: {FOLDER_STRUCTURE} not existing'
#     assert len(os.listdir(FOLDER_STRUCTURE) ) > 0,  f'folder: {FOLDER_STRUCTURE} not containing files'

#     for i, file in enumerate(parse_folder(FOLDER_STRUCTURE)):
        
#         filenames.append(file)
#         df= pd.read_csv(file)
#         dict_dfs[file] = df 
#         if small_part and i > 100:
#             break

#     print(' Conversion of categorical values')


#     def inverse_mapping(mapping):
#         inverse = {}
#         for key in mapping:
#             for value in mapping[key]:
#                 inverse[value] = key
#         return inverse
#     all_mappings_inv={col:inverse_mapping(all_mappings[col]) for col in all_mappings.keys()}
#     all_mappings_inv

#     #### 2m
#     print('convert_values_of_categorical_columns')
#     def convert_values_of_categorical_columns(df):
#         # replace the wrong values with the correct ones
#         df.replace(all_mappings_inv, inplace=True)
#         categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
#         for col in categorical_columns:
#             #get from 1 XXX to -> 1
#             df[col] = df[col].str.split(" ").apply(
#                 lambda x: x[0] if isinstance(x, list) else x)
#             # convert categorical column to numeric
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         return df
        

#     dict_dfs_processed = {}
#     for k, df in dict_dfs.items():

#         res = convert_values_of_categorical_columns(df.copy())
#         if res is None:
#             print(k)
#             break
#         dict_dfs_processed[k]= res


#     # dict_dfs_processed = pickle.load(open('dict_dfs_processed.pk', 'rb'))

#     print(' 3m create categories per column & df')
#     dict_dfs_cat = {}
#     for k, df in tqdm(dict_dfs_processed.items()):
#         df = df.copy()
#         df.drop('Hours', axis=1, inplace=True)#.shape

#         if (df['Height'] is not None) and (df['Weight'] is not None):   
#             df['Height'].fillna(method='ffill', inplace=True)
#             df['Height'].fillna(method='bfill', inplace=True)
            
#             df['BMI']=(df['Weight']/((df['Height']/100)**2))
#         else:
#             df['BMI']=math.nan
#         df.drop(['Height', 'Weight', 'Fraction inspired oxygen'], axis=1, inplace=True)
        
#         for col in rules.keys():
                
#             arr1 = df[col].apply(lambda x: [rule(x) for rule in rules[col]])
#             arr = arr1.apply(lambda x: x.index(True) if True in x else None)
#             df[col]=arr

#         dict_dfs_cat[k] = df.apply(lambda x:x.unique())
#     pickle.dump(dict_dfs_cat, open('dict_dfs_cat.pk', 'wb'))
#     # if all the values are nan then add normal category

# print(f'Phase 2')
# fname= f'alldfs_categories{spstr}.csv'

# if not os.path.exists(fname):
#     dict_dfs_cat = pickle.load(open('dict_dfs_cat.pk', 'rb'))
#     print('### remove nans and Nones')
#     dict_dfs_cat_uniform = {}
#     for k, series in tqdm(dict_dfs_cat.items()):
        
#         series = series.apply(lambda x: sorted(x))
#         # filter nan values from lists in series cells
#         series = series.apply(lambda x: [i for i in x if i is not None])
#         try:
#             if len(series) > 1:
#                 series = series.apply(lambda x: [i for i in x if not math.isnan(i)])
                
#         except:
#             import pdb; pdb.set_trace() 
#         dict_dfs_cat_uniform[k] = series

#     alldfs = pd.DataFrame.from_dict(dict_dfs_cat_uniform, orient='index',columns = (list(dict_dfs_cat_uniform.values())[0]).index.tolist())
#     alldfs.to_csv(fname)
        


print(f'Phase 3')
fname = f'A_expert_edges_exact{spstr}.pk'
if not os.path.exists(fname):
    # graph_common_anomalies = False
    # if graph_common_anomalies:
    alldfs= pd.read_csv(f'alldfs_categories{spstr}.csv')
    

    alldfs=alldfs.set_index(['Unnamed: 0'])


    # select cells of dataframe  whocse  value is empty pd.Series
    alldfs=alldfs.applymap(lambda x: '[]' if (type(x) == str and 'Series' in x) else x)
    alldfs=alldfs.applymap(lambda x: eval(x))

    # for all cells in df if it is in normal category then replace it with nan
    all_normal_categories = [rules_normal[col] for col in rules_normal.keys()]
    alldfs=alldfs.applymap(lambda x: [] if x == [all_normal_categories]  else x)

    # replace nan with empty list
    alldfs=alldfs.applymap(lambda x: [] if x == [math.nan]  else x)

    print(' Multi hot encode')
    # new dataframe with multiple columns one for each dist column of alldfs dataframe
    alldfs_multi_hot = pd.DataFrame()
    alldfs_multi_hot['id'] = alldfs.index
    for col in alldfs.columns:
        alldfs_multi_hot[col] = alldfs[col].apply(lambda x: [0 for i in range(len(rules[col]))] if len(x) == 0 else x).tolist()
        alldfs_multi_hot[col] = alldfs_multi_hot[col].apply(lambda x: [1 if i in x else 0 for i in range(len(rules[col]))]).tolist()
        for i in range(len(rules[col])):
            alldfs_multi_hot[col+'_'+str(i)] = alldfs_multi_hot[col].apply(lambda x: x[i])
        alldfs_multi_hot.drop(col, axis=1, inplace=True)
        # import pdb; pdb.set_trace()
    alldfs_multi_hot.set_index('id', inplace=True)


    import datetime
    print('Calculate exact adjacency between episodes')
    print('Current time: ', datetime.datetime.now())

    gs= alldfs_multi_hot.groupby(list(alldfs_multi_hot.columns))
    clusters = list(gs.groups.values())
    expert_edges = pd.Series(clusters).apply(lambda x : [x for x in it.combinations(x, 2)])
    pickle.dump(expert_edges, open(fname, 'wb'))
    print(f'Saved: {fname}')

print(f'Ar = {fname} loaded')
Ar = pickle.load(open(fname, 'rb'))
