from torch_geometric.data import Data

import torch_geometric 
import itertools as iter
import os
import pdb
import pickle
import random

# edge_strategies.py
import string

import numpy as np
import pandas as pd
import torch

make_random_edges = lambda n_nodes, n_edges: random.choices(
    list(iter.combinations(range(n_nodes), 2)), k=n_edges
)

no_edges = lambda n_nodes, n_edges: []

fully_connected = lambda n_nodes, n_edges: list(iter.combinations(range(n_nodes), 2))

commorbidity_edges = []

common_same_patient_edges = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def read_csv_as_string(filename):
    with open(filename, "r") as f:
        return f.read()


def regex_replace_with_dict(s: str, d):
    for k, v in d.items():
        s = s.replace(k, v)
    return s


def save_lines_as_csv(lines, filename):
    # pdb.set_trace()
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def save_string_as_csv(s, filename):
    with open(filename, "w") as f:
        f.write(s)


def inverse_dict(d):
    return {v: k for k, v in d.items()}


# def k_nn_kde_graph(node_features, k=100, max_edges= None):
#     """Make edges between k nearest neighbours in node_features"""
#     from sklearn.neighbors import NearestNeighbors
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='kde').fit(node_features)
#     distances, indices = nbrs.kneighbors(node_features)
#     computed_graph =  list(iter.chain.from_iterable([list(iter.combinations(x, 2)) for x in indices]))
#     if max_edges is not None:
#         computed_graph = computed_graph[:max_edges]
#     return computed_graph

# def k_nn_ballTree_graph(node_features, k=100, max_edges= None):
#     """Make edges between k nearest neighbours in node_features"""
#     from sklearn.neighbors import NearestNeighbors
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(node_features)
#     distances, indices = nbrs.kneighbors(node_features)
#     computed_graph =  list(iter.chain.from_iterable([list(iter.combinations(x, 2)) for x in indices]))
#     if max_edges is not None:
#         computed_graph = computed_graph[:max_edges]
#     return computed_graph


from torch_geometric.transforms import KNNGraph, RadiusGraph


# def expert_edges(train_names):
#     pass


# def radius_graph(node_features, radius=0.1, max_edges=None):
#     """Make edges between k nearest neighbours in node_features"""
#     computed_graph = RadiusGraph(radius=radius).transform(node_features)
#     if max_edges is not None:
#         computed_graph = computed_graph[:max_edges]
#     return computed_graph
def weighted_knn_graph(node_features, loop=True, distance_cosine=True, k=100, num_workers=5):
    global device
    edge_index = torch_geometric.nn.knn_graph(
        node_features,
        k=k,
        loop=loop,
        num_workers=num_workers,
    )
    distances = torch.cdist(node_features, node_features)
    weights = distances[edge_index[0], edge_index[1]]

    new_data = Data(pos=torch.tensor(node_features))
    new_data.to(device=device)
    result_graph = self.computed_graph(new_data)

    return result_graph

class knn_graph:
    def __init__(
        self, node_features, loop=True, distance_cosine=True, k=100, num_workers=5
    ):
        """Make edges between k nearest neighbours in node_features"""
        self.computed_graph = KNNGraph(
            k=k,
            loop=True,
            force_undirected=False
            # , cosine=distance_cosine
        )  # .transform(node_features)

    def __call__(self, node_features):
        global device
        from torch_geometric.data import Data

        new_data = Data(pos=torch.tensor(node_features))
        new_data.to(device=device)
        # import pdb; pdb.set_trace()
        result_graph = self.computed_graph(new_data)

        return result_graph


def feature_anomaly_edges_automatic(train_name):
    """Make connections according to index of train_name and analysis of raw node_features"""
    if os.path.exists("091022_all_edges.pkl"):
        return pickle.load(open("091022_all_edges.pkl", "rb"))

    if os.path.exists("converted.csv"):
        df = pd.read_csv("converted.csv")
    else:
        correct_indexes_dict_raw = pd.Series(train_name).to_dict()
        correct_indexes_dict_converted = {
            str(k): v for k, v in correct_indexes_dict_raw.items()
        }
        correct_indexes_dict_converted_inverted = inverse_dict(
            correct_indexes_dict_converted
        )

        ANALYSIS_RESULTS_FILE_PATH = "2022-10-06_17-10-26______episodes_per_group_4_bins__Oxygen saturation : Temperature : Systolic blood pressure : Diastolic blood pressure.csv"
        df = pd.read_csv(ANALYSIS_RESULTS_FILE_PATH)
        for k, v in correct_indexes_dict_converted_inverted.items():
            df["episodes_per_group"].replace(regex="'" + k, value="'" + v, inplace=True)

        print("Generating file...")
        df.to_csv("converted.csv")

    result = df.apply(
        lambda x: iter.combinations(eval(x["episodes_per_group"]), 2), axis=1
    )
    all_edges = [edge_pair for edge_iterator in result for edge_pair in edge_iterator]
    pickle.dump(all_edges, open("091022_all_edges.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    return all_edges


spearman_correlation_edges = []


def find_duplicate_tuples_in_list(l):
    return [x for n, x in enumerate(l) if x in l[:n]]


edge_strategies = {"knn_graph": knn_graph, "quantile": feature_anomaly_edges_automatic}

if __name__ == "__main__":
    from create_homogeneous_graph_4 import (
        get_most_recent_file,
        read_list_pickle_files_parallel,
    )

    list_files = get_most_recent_file()
    list_files = [list_files[1], list_files[7], list_files[13]]
    # pdb.set_trace()
    l = read_list_pickle_files_parallel(list_files)

    all = np.concatenate([np.concatenate(e) for e in l])
    assert len(
        pd.Series(all).value_counts()[pd.Series(all).value_counts() == 1]
    ) == len(all)
    # train_embedding=np.concatenate(l[0])
    train_name = np.concatenate(l[0])
    val_name = np.concatenate(l[1])
    test_name = np.concatenate(l[2])

    all_names = np.concatenate([train_name, val_name, test_name])

    edges = feature_anomaly_edges_automatic(all_names)
    # pdb.set_trace()
    # print(spearman_correlation_edges(10, 5))
    # print(feature_anomaly_edges(10, 5))

    # print(feature_anomaly_edges_automatic(10, 5))

    # print(spearman_correlation_edges(10, 5))

    # print(feature_anomaly_edges(10, 5))

    # print(feature_anomaly_edges_automatic(10, 5))

    # print(spearman_correlation
