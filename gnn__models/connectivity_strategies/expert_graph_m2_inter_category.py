import datetime 
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
counter = 0 
counter +=1 
print(f'Phase {counter}')
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
import itertools
def apply_logical_and_axis_1_to_df(df):
    df = df.copy()
    df = df.apply(lambda x: list(itertools.combinations(x, 2)))
    df = df.apply(lambda x: [i for i in x if i[0] == i[1]])
    df = df.apply(lambda x: [i[0] for i in x])
    return df
    


counter +=1 
print(f'Phase {counter}')

fname = f'A_m2_expert_edges_inter_category{spstr}.pk'
if not os.path.exists(fname):
    # graph_common_anomalies = False
    # if graph_common_anomalies:
    alldfs= pd.read_csv(f'alldfs_categories{spstr}.csv')
    

    alldfs=alldfs.set_index(['Unnamed: 0'])

    # select cells of dataframe  whocse  value is empty pd.Series
    alldfs=alldfs.applymap(lambda x: '[]' if (type(x) == str and 'Series' in x) else x)
    alldfs=alldfs.applymap(lambda x: eval(x))

    ## alldfs when cell is []  then fill with column normal values 
    alldfs = alldfs.apply(lambda x: x.apply(lambda y: [rules_normal[x.name]] if y == [] else y))
    # alldfs1 = alldfs.apply(lambda x:  rules_normal[x.name], x[0])
    
    # for all cells in df if it is in normal category
    # all_normal_categories = [rules_normal[col] for col in rules_normal.keys()]
    # alldfs=alldfs.applymap(lambda x: [] if x == [all_normal_categories]  else x)

    # replace nan with empty list
    # alldfs=alldfs.applymap(lambda x: [] if x == [math.nan]  else x)

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
    
    # link all healthy patients to each other
    multihotcols_grouped_by_alldfs_cols = {col:[col+'_'+str(i) for i in range(len(rules[col]))] for col in alldfs.columns}
    normal_columns = [col + '_' + str(rules_normal[col]) for col in alldfs.keys()]
    abnormal_columns = [i for i in alldfs_multi_hot.columns if i not in normal_columns]
    multihotcols_grouped_by_alldfs_abnormal_cols = {}
    for col in abnormal_columns:
        multihotcols_grouped_by_alldfs_abnormal_cols[col.split('_')[0]] = []

    for col in abnormal_columns:
        multihotcols_grouped_by_alldfs_abnormal_cols[col.split('_')[0]].append(col)
    
    
    df = (alldfs_multi_hot[abnormal_columns] == 0)
    df['and'] = df[df.columns[0]]
    for col in df.columns[1:]: 
        df['and'] = df['and'] & df[col]
    healthy_episode_links = [x for x in it.combinations(df[df['and']].index, 2)]
    
    # apply accros column  logical OR 
    alldfs_multi_hot.loc[:, abnormal_columns] = alldfs_multi_hot.loc[:, abnormal_columns].apply(lambda row: row == 0, axis=1)
    for key,cols in multihotcols_grouped_by_alldfs_abnormal_cols.items():
        alldfs_multi_hot[f'{key}_or'] = alldfs_multi_hot[cols[0]]
        for col in cols: 
            print(f'key: {key} col: {col}')
            alldfs_multi_hot[f'{key}_or'] = alldfs_multi_hot[f'{key}_or'] | alldfs_multi_hot[col]
    
    # copy to new df the  columns containing '_or'
    df = alldfs_multi_hot[[col for col in alldfs_multi_hot.columns if '_or' in col]]
    
    clusters = df.groupby(df.columns.tolist(),as_index=False) # .size())    
    # import pdb; pdb.set_trace()
    # df[df['and'] == True]
    # import pdb; pdb.set_trace()
    # df
    print('Current time: ', datetime.datetime.now())
    
    expert_edges = pd.Series(clusters.groups.values()).apply(lambda x : [x for x in it.combinations(x, 2)])
    
    expert_edges_filtered = [edges for edges in expert_edges if edges !=[]]
    expert_edges = list(it.chain.from_iterable(expert_edges_filtered))
    # healthy_episode_links # add
    expert_edges = expert_edges + healthy_episode_links
    print('End time: ', datetime.datetime.now())
    
    
    pickle.dump(expert_edges, open(fname, 'wb'))
    print(f'Saved: {fname}')
    # import pdb; pdb.set_trace()

print(f'Ar = {fname} loaded')
Ar = pickle.load(open(fname, 'rb'))

