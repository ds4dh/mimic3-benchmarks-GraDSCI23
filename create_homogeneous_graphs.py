from torch.utils.data import DataLoader, random_split, Dataset, Subset
import argparse
import glob
import itertools
import os
import pickle
import random
import re
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from edge_strategies import knn_graph
from read_embeddings import take_all_features 


DATE_TIME=datetime.now().strftime("%d%m%Y-%H%M%S")

def read_pickle_file(pickle_file):
    """Read pickle file."""
    print(pickle_file)
    with open(pickle_file, "rb") as f:
        fcontent = pickle.load(f)
        f.close()
        return fcontent


def read_list_pickle_files_parallel(list_pickle_files, n_processes=4):
    """Read list of pickle files in parallel."""
    res = []
    print("Reading pickle files in parallel")
    for i in range(len(list_pickle_files)):
        # with Pool(n_processes) as p:
        res.append(read_pickle_file(list_pickle_files[i]))
    # import pdb; pdb.set_trace()
    return res


def convert_list_tuple_string_to_int(list_tuple_string):
    return [(int(l[0]), int(l[1])) for l in list_tuple_string]



def get_most_recent_file(folder=""):
    """Get most recent file."""

    list_files = glob.glob(os.path.join(folder, "*.pkl"))  # what should these files be?

    is_datetime_in_file_name = (
        lambda x: re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", x) is not None
    )

    print("list_files", list_files)
    parent_folder = os.path.dirname(folder)
    print("parent folder", parent_folder)
    list_files = [f for f in list_files if is_datetime_in_file_name(f)]

    find_timestamp_in_file_name = lambda x: datetime.strptime(
        re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", x).group(),
        "%Y-%m-%dT%H:%M:%S",
    )
    last_timestamp_among_files = max(
        [find_timestamp_in_file_name(f) for f in list_files]
    )
    string_last_timestamp_among_files = last_timestamp_among_files.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    list_files = [x for x in list_files if string_last_timestamp_among_files in x]

    return list_files


def read_lstm_embeddings(folder=""):
    print("Reading lstm embeddings from folder")

    file_list = [
        "16_11_2022/test_embedding_hidden_embedding_2022-11-16T23:48:34.pkl",
        "16_11_2022/val_embedding_embedding_2022-11-16T23:48:34.pkl",
        "16_11_2022/test_embedding_embedding_2022-11-16T23:48:34.pkl",
        "16_11_2022/train_embedding_input_2022-11-16T23:48:34.pkl",
        "16_11_2022/test_embedding_ys_2022-11-16T23:48:34.pkl",
        "16_11_2022/train_embedding_ys_2022-11-16T23:48:34.pkl",
        "16_11_2022/val_embedding_predictions_2022-11-16T23:48:34.pkl",
        "16_11_2022/test_embedding_ts_2022-11-16T23:48:34.pkl",
        "16_11_2022/train_embedding_ts_2022-11-16T23:48:34.pkl",
        "16_11_2022/val_embedding_hidden_embedding_2022-11-16T23:48:34.pkl",
        "16_11_2022/test_embedding_predictions_2022-11-16T23:48:34.pkl",
        "16_11_2022/test_embedding_name_2022-11-16T23:48:34.pkl",
        "16_11_2022/val_embedding_ts_2022-11-16T23:48:34.pkl",
        "16_11_2022/train_embedding_predictions_2022-11-16T23:48:34.pkl",
        "16_11_2022/val_embedding_input_2022-11-16T23:48:34.pkl",
        "16_11_2022/val_embedding_name_2022-11-16T23:48:34.pkl",
        "16_11_2022/test_embedding_input_2022-11-16T23:48:34.pkl",
        "16_11_2022/train_embedding_name_2022-11-16T23:48:34.pkl",
        "16_11_2022/val_embedding_ys_2022-11-16T23:48:34.pkl",
        "16_11_2022/train_embedding_embedding_2022-11-16T23:48:34.pkl",
        "16_11_2022/train_embedding_hidden_embedding_2022-11-16T23:48:34.pkl",
    ]

    res = {}
    for i in range(len(file_list)):
        # with Pool(n_processes) as p:
        res[file_list[i]] = read_pickle_file(file_list[i])

    train_embedding = np.concatenate(
        res["16_11_2022/train_embedding_hidden_embedding_2022-11-16T23:48:34.pkl"]
    )
    train_name = np.concatenate(
        res["16_11_2022/train_embedding_name_2022-11-16T23:48:34.pkl"]
    )
    train_predictions = np.concatenate(
        res["16_11_2022/train_embedding_predictions_2022-11-16T23:48:34.pkl"]
    )
    train_input = np.concatenate(
        res["16_11_2022/train_embedding_input_2022-11-16T23:48:34.pkl"]
    )
    train_ts = np.concatenate(
        res["16_11_2022/train_embedding_ts_2022-11-16T23:48:34.pkl"]
    )
    train_ys = np.concatenate(
        res["16_11_2022/train_embedding_ys_2022-11-16T23:48:34.pkl"]
    )

    val_embedding = np.concatenate(
        res["16_11_2022/val_embedding_hidden_embedding_2022-11-16T23:48:34.pkl"]
    )
    val_name = np.concatenate(
        res["16_11_2022/val_embedding_name_2022-11-16T23:48:34.pkl"]
    )
    val_predictions = np.concatenate(
        res["16_11_2022/val_embedding_predictions_2022-11-16T23:48:34.pkl"]
    )
    val_input = np.concatenate(
        res["16_11_2022/val_embedding_input_2022-11-16T23:48:34.pkl"]
    )
    val_ts = np.concatenate(res["16_11_2022/val_embedding_ts_2022-11-16T23:48:34.pkl"])
    val_ys = np.concatenate(res["16_11_2022/val_embedding_ys_2022-11-16T23:48:34.pkl"])

    test_embedding = np.concatenate(
        res["16_11_2022/test_embedding_hidden_embedding_2022-11-16T23:48:34.pkl"]
    )
    test_name = np.concatenate(
        res["16_11_2022/test_embedding_name_2022-11-16T23:48:34.pkl"]
    )
    test_predictions = np.concatenate(
        res["16_11_2022/test_embedding_predictions_2022-11-16T23:48:34.pkl"]
    )
    test_input = np.concatenate(
        res["16_11_2022/test_embedding_input_2022-11-16T23:48:34.pkl"]
    )
    test_ts = np.concatenate(
        res["16_11_2022/test_embedding_ts_2022-11-16T23:48:34.pkl"]
    )
    test_ys = np.concatenate(
        res["16_11_2022/test_embedding_ys_2022-11-16T23:48:34.pkl"]
    )

    return {
        "train_embedding": train_embedding,
        "train_name": train_name,
        "train_predictions": train_predictions,
        "train_input": train_input,
        "train_ts": train_ts,
        "train_ys": train_ys,
        "val_embedding": val_embedding,
        "val_name": val_name,
        "val_predictions": val_predictions,
        "val_input": val_input,
        "val_ts": val_ts,
        "val_ys": val_ys,
        "test_embedding": test_embedding,
        "test_name": test_name,
        "test_predictions": test_predictions,
        "test_input": test_input,
        "test_ts": test_ts,
        "test_ys": test_ys,
    }


def shortened_path(data, train_df, val_df, test_df):
    counter_time_execution = datetime.now()
    all_data = train_df.index.tolist() + val_df.index.tolist() + test_df.index.tolist()
    mapping_index_name = pd.Series(all_data).to_dict()
    mapping_name_index = {v: k for k, v in mapping_index_name.items()}

    print("Loaded ! Time execution: ", datetime.now() - counter_time_execution)
    if type(data) == pd.Series:
        data1 = [item for sublist in data for item in sublist]
        print(
            "Data: list of groups I guess! Time execution: ",
            datetime.now() - counter_time_execution,
        )
        data2 = list(set(data1))
    else:
        data2 = data
    print("len(data2): ", len(data2))
    print(
        "Regex findall! Time execution: ",
        datetime.now() - counter_time_execution,
    )
    extracted_strings = re.findall(r"\/([^\/]*_timeseries\.csv)", str(data2))
    shorted_node_names = [
        tuple(extracted_strings[i : i + 2]) for i in range(0, len(extracted_strings), 2)
    ]
    assert type(shorted_node_names[0][0]) == str, "should be string"
    edges = [
        (mapping_name_index[v[0]], mapping_name_index[v[1]])
        for v in tqdm(shorted_node_names)
    ]
    assert type(edges) == list, "should be list"
    assert type(edges[0][0]) == int, "should be int"
    return edges

class MyOwnDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        node_type_embeddings="lstm",
        edge_strategy_name="expert_exact",
        k=10,
        n_edges=None,
        distance_euclidean=True,
        transform=None,
        pre_transform=None,
        pre_filter=None, 
        device="cpu",
    ):
        self.edge_strategy_name = edge_strategy_name
        self.k = k
        self.n_edges = n_edges
        self.distance_euclidean = distance_euclidean
        self.node_type_embeddings = node_type_embeddings
        super().__init__(root, transform, pre_transform, pre_filter)
        # import pdb; pdb.set_trace()
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=torch.device(device))
        # import pdb; pdb.set_trace()

    def raw_file_names(self):
        return ["some_file_1", "some_file_2"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        train_df, val_df, test_df = take_all_features()
        # assert len(train_df) > len(val_df) and len(val_df) > len(test_df), "Train, val, test dataframes are not Correct sizes (train > val > test)"
        self.node_embeddings = (train_df, val_df, test_df)
        assert (len(train_df.index.tolist() + val_df.index.tolist() + test_df.index.tolist())> 0), "should be more than 0"

        print("Read Nodes ")
        train_df, val_df, test_df = self.node_embeddings

        if self.node_type_embeddings == "lstm":
            train_X = train_df["lstm_embedding"]
            val_X = val_df["lstm_embedding"]
            test_X = test_df["lstm_embedding"]
            train_y = train_df["ys"]
            val_y = val_df["ys"]
            test_y = test_df["ys"]
        if self.node_type_embeddings == "grnn":
            train_X = train_df["grnn_embedding"]
            val_X = val_df["grnn_embedding"]
            test_X = test_df["grnn_embedding"]
            train_y = train_df["ys"]
            val_y = val_df["ys"]
            test_y = test_df["ys"]
        if self.node_type_embeddings == "stat":
            train_X = train_df["stat_features"]
            val_X = val_df["stat_features"]
            test_X = test_df["stat_features"]
            train_y = train_df["ys"]
            val_y = val_df["ys"]
            test_y = test_df["ys"]

        X_all = torch.cat(
            (
                torch.Tensor(train_X),
                torch.Tensor(val_X),
                torch.Tensor(test_X),
            ),
            0,
        )

        print("Create edges")

        assert  self.edge_strategy_name in ['random', 'expert_exact', 'expert_medium', 'expert_lenient', 'knn_graph', 'trivial'], 'should be one of the three'
        if "expert" in self.edge_strategy_name:
            assert self.edge_strategy_name in ['expert_exact', 'expert_medium', 'expert_lenient'], 'should be one of the three'
            PATH = "."
            EDGE_FILES = {
                "expert_exact": "A_expert_edges_exact.pk",
                "expert_medium": "A_m2_expert_edges_inter_category.pk",
                "expert_lenient": "A_m3_expert_edges_inter_category.pk",
            }
            data = pickle.load(open(os.path.join(PATH, EDGE_FILES[self.edge_strategy_name]), "rb"))
            tensor_edges = torch.tensor(shortened_path(data, train_df, val_df, test_df))
            tensor_edges = torch.swapaxes(tensor_edges, 1, 0)
            edge_index_mod_self, edge_attr = add_self_loops(tensor_edges)
            graph = Data(
                x=X_all,
                edge_index=edge_index_mod_self,
            )
     
        if self.edge_strategy_name == "random":
            # edge_strategy = feature_anomaly_edges_automatic
            make_random_edges = lambda n_nodes, n_edges: random.choices(
                list(itertools.combinations(range(n_nodes), 2)), k=self.n_edges
            )
            # import pdb; pdb.set_trace()
            A = make_random_edges(len(X_all), self.n_edges)
            # import pdb; pdb.set_trace()
            # edge_index_mod = convert_list_tuple_string_to_int(A)
            edge_index_torch = torch.tensor(A).T
            edge_index_mod_self, edge_attr = add_self_loops(edge_index_torch)
            graph = Data(
                x=X_all,
                edge_index=edge_index_mod_self,
            )

        if self.edge_strategy_name == "knn_graph":
            edge_strategy = knn_graph(
                train_X, k=self.k, loop=False, distance_cosine=self.distance_euclidean
            )
            graph = edge_strategy(X_all)
            graph.x = graph.pos
            del graph.pos

        if self.edge_strategy_name == "trivial":
            edge_index = torch.empty((2, 0), dtype=torch.long)
            graph = Data(x=X_all, edge_index=edge_index,)
            num_nodes = X_all.size(0)
            edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            graph.edge_index = edge_index

        print("Create labels ")
        all_Y = torch.cat([torch.tensor(train_y), torch.tensor(val_y), torch.tensor(test_y)])
        train_mask = torch.cat([torch.ones(len(train_X)),torch.zeros(len(val_X)),torch.zeros(len(test_X)),],0,)
        val_mask = torch.cat([torch.zeros(len(train_X)),torch.ones(len(val_X)),torch.zeros(len(test_X)),],0,)
        test_mask = torch.cat([torch.zeros(len(train_X)),torch.zeros(len(val_X)),torch.ones(len(test_X)),],0,)

        graph.y = all_Y
        graph.train_mask = train_mask.bool()
        graph.val_mask = val_mask.bool()
        graph.test_mask = test_mask.bool()
        data_list = [graph]
        

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # import pdb; pdb.set_trace()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Processed data to: ", self.processed_paths[0])


if __name__ == "__main__":
    # parse argument edge_strategy_name with values allowed: 'quantile', 'knn_graph'
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_strategy_name", type=str, required=True)
    parser.add_argument("--node_embeddings_type", type=str, required=True)  # lstm_pca_6 # lstm_pca50
    parser.add_argument("--k", type=int, default="10")
    parser.add_argument("--n_edges", type=int, default=300_000)
    parser.add_argument("--folder_name", type=str, required=True)
    
    # add boolean argument to the parser and store false if specified
    parser.add_argument("--distance_euclidean", action="store_false")

    print("Arguments: ", parser.parse_args())

    # parser.add_argument('--distance_euclidean', type=bool, default=True, action='store_false')
    args = parser.parse_args()
    edge_strategy_name = args.edge_strategy_name
    assert  edge_strategy_name in ['random', 'expert_exact', 'expert_medium', 'expert_lenient', 'knn_graph', 'trivial'], 'should be one of the three'
    k = args.k
    distance_euclidean_else_cosine = args.distance_euclidean
    name_distance = "cosine" if distance_euclidean_else_cosine else "euclidean"
    node_embeddings_type = args.node_embeddings_type
    folder_name = args.folder_name
    assert folder_name != None and folder_name != '' , "folder_name should be specified"
    folder_name = f"{folder_name}/data_{edge_strategy_name}_{node_embeddings_type}"

    print('SAVING folder_name: ', folder_name)
    dataset = MyOwnDataset(
        root=folder_name,
        node_type_embeddings=node_embeddings_type,
        edge_strategy_name=edge_strategy_name,
        k=k,  # for knn_graph only
        n_edges=args.n_edges,  # for random only
    )
    data = dataset[0]
    print("Created file: ")
    FOLDER_NAME = f'data_{edge_strategy_name}_k{str(k)}_{name_distance}_{node_embeddings_type.replace(".","_")}'

    def append_to_file(file_name, text_to_append):
        """Append given text as a new line at the end of file"""
        with open(file_name, "a+") as file_object:
            file_object.seek(0)
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("")
            file_object.write(text_to_append)

    report_name = "data_creation" + DATE_TIME + FOLDER_NAME
    NUMBER_NODES = data.x.shape[0]
    NUMBER_EDGES = data.edge_index.shape[1]
    append_to_file(
        "graph_creation.txt",
        f"{report_name}: Number of nodes: {NUMBER_NODES} \t Number of edges: {NUMBER_EDGES} \n",
    )

# python create_homogeneous_graph_4.py --edge_strategy expert_exact --node_embeddings_type stat 
# python create_homogeneous_graph_4.py --edge_strategy expert_medium --node_embeddings_type stat 
# python create_homogeneous_graph_4.py --edge_strategy expert_lenient --node_embeddings_type stat 
# python create_homogeneous_graph_4.py --edge_strategy random  --node_embeddings_type stat 
# python create_homogeneous_graph_4.py --edge_strategy knn_graph --node_embeddings_type stat 
# python create_homogeneous_graph_4.py --edge_strategy trivial --node_embeddings_type stat 

# python create_homogeneous_graph_4.py --edge_strategy expert_exact --node_embeddings_type lstm 
# python create_homogeneous_graph_4.py --edge_strategy expert_medium --node_embeddings_type lstm 
# python create_homogeneous_graph_4.py --edge_strategy expert_lenient --node_embeddings_type lstm 
# python create_homogeneous_graph_4.py --edge_strategy random  --node_embeddings_type lstm 
# python create_homogeneous_graph_4.py --edge_strategy knn_graph --node_embeddings_type lstm 
# python create_homogeneous_graph_4.py --edge_strategy trivial --node_embeddings_type lstm 