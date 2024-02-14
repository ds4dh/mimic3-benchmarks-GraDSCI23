# import pickle data  folder  data/phenotyping/statistical_features 
# ls data/phenotyping/statistical_features 
# test_X          test_ts         train_X         train_ts        val_X           val_ts
# test_names      test_y          train_names     train_y         val_names       val_y

import os 
import pandas as pd 
import torch
import pickle 
import numpy as np 

def read_pickle_file(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

def read_statistical_features(FOLDER_PATH='data/phenotyping/statistical_features'):
    data = {}
    prefixes = ['train', 'val', 'test']
    suffixes = ['names', 'ts', 'X', 'y']
    
    for prefix in prefixes:
        for suffix in suffixes:
            data[f"{prefix}_{suffix}"] = read_pickle_file(os.path.join(FOLDER_PATH, f"{prefix}_{suffix}"))
    
    dfs = {}
    for prefix in prefixes:
        df = pd.DataFrame({
            'stat_features': data[f"{prefix}_X"].tolist(),
            'name': data[f"{prefix}_names"],
            'y': data[f"{prefix}_y"].tolist()
        })
        df.set_index('name', inplace=True)
        dfs[prefix] = df

    return dfs['train'], dfs['val'], dfs['test']


def read_lstm_embeddings1(folder):
    print('Reading lstm embeddings from folder')
    data = {}
    for file in os.listdir(folder):
        data[file] = np.load(os.path.join(folder, file))
    # map 
    file_to_variable_mapping = {
        "hidden_hn_train.npy": "train_embedding",
        "hidden_hn_val.npy": "val_embedding",
        "hidden_hn_test.npy": "test_embedding",
        "labels_train.npy": "train_ys",
        "labels_val.npy": "val_ys",
        "labels_test.npy": "test_ys",
        "predictions_train.npy": "train_predictions",
        "predictions_val.npy": "val_predictions",
        "predictions_test.npy": "test_predictions",
        "name_train.npy": "train_name",
        "name_val.npy": "val_name",
        "name_test.npy": "test_name",
    }
    
    # map the file to the variable
    to_return = {}
    for file, variable in file_to_variable_mapping.items():
        to_return[variable] = data[file]

    return to_return


def read_lstm_embeddings(folder):
    data=read_lstm_embeddings1(folder)
    train_cols = [k for k in data.keys() if 'train' in k and 'input' not in k]
    # assert all first dimension is the same
    assert all([data[k].shape[0] == data[train_cols[0]].shape[0] for k in train_cols])
    # assert all second dimension is the same
    # assert all([data[k].shape[1] == data[train_cols[0]].shape[1] for k in train_cols])

    #     .keys()
    # dict_keys(['train_embedding', 'val_embedding', 'test_embedding', 'train_ys', 'val_ys', 'test_ys', 'train_predictions', 'val_predictions', 'test_predictions'])
    # (Pdb) 
    train_df = pd.DataFrame({'name':data['train_name'],'ys':data['train_ys'].tolist(),'lstm_embedding':data['train_embedding'].tolist()})
    train_df.head()

    # use name column  as index 
    train_df.set_index('name', inplace=True)
    assert len(train_df.loc[train_df['ys'].isna()]) == 0

    val_cols = [k for k in data.keys() if 'val' in k and 'input' not in k]
    # assert all first dimension is the same
    assert all([data[k].shape[0] == data[val_cols[0]].shape[0] for k in val_cols])

    val_df = pd.DataFrame({
        'name':data['val_name'],
        # 'ts':data['val_ts'],
        'ys':data['val_ys'].tolist(),
        # 'predictions':data['val_predictions'],
        'lstm_embedding':data['val_embedding'].tolist()
    })
    val_df.head 
    val_df.set_index('name', inplace=True)

    test_cols = [k for k in data.keys() if 'test' in k and 'input' not in k]
    # assert all first dimension is the same
    assert all([data[k].shape[0] == data[test_cols[0]].shape[0] for k in test_cols])

    test_df = pd.DataFrame({
        'name':data['test_name'],
        # 'ts':data['test_ts'],
        'ys':data['test_ys'].tolist(),
        # 'predictions':data['test_predictions'],
        'lstm_embedding':data['test_embedding'].tolist()
    })
    test_df.head() 
    test_df.set_index('name', inplace=True)

    return train_df, val_df, test_df


def take_all_features():
    stat_train_df, stat_val_df, stat_test_df = read_statistical_features()
    lstm_train_df, lstm_val_df, lstm_test_df = read_lstm_embeddings('data/BCE_LSTM_2024-02-12/')
    # mgrn_train_df, mgrn_val_df, mgrn_test_df = read_mgrnn_embeddings()
    train_df = pd.concat([stat_train_df, lstm_train_df], axis=1)
    val_df = pd.concat([stat_val_df, lstm_val_df], axis=1)
    test_df = pd.concat([stat_test_df, lstm_test_df], axis=1)
    assert len(train_df) == len(stat_train_df) == len(lstm_train_df)
    assert len(val_df) == len(stat_val_df) == len(lstm_val_df)
    assert len(test_df) == len(stat_test_df) == len(lstm_test_df)
    # assert len(train_df) > len(val_df) and len(val_df) > len(test_df), "Train, val, test dataframes are not Correct sizes (train > val > test) but instead: train: {}, val: {}, test: {}".format(len(train_df), len(val_df), len(test_df))
    assert all([df[col].isna().sum() == 0 for df in [train_df, val_df, test_df] for col in df.columns]), "There are NaNs in the dataframes"
    
    return train_df, val_df, test_df

if __name__ == '__main__':
    # unit test read_statistical_features data/phenotyping/statistical_features 
    train_df, val_df, test_df = read_statistical_features()
    # describe the data
    print(train_df.describe())
    print(val_df.describe())
    print(test_df.describe())

    test_df, train_df, val_df = take_all_features()

    print(train_df.describe())
    print(val_df.describe())
    print(test_df.describe())
