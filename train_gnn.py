from tqdm import tqdm 
import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm
from torch_geometric.data import DataLoader, Data, Dataset

from create_homogeneous_graphs import MyOwnDataset
from custom_layers import activation_name_implementation, get_layer_impl_from_layer_type
from torch_geometric.utils import subgraph

def get_loss(y_pre, labels, alpha=0):
    criterion_BCE = nn.BCELoss()
    loss = criterion_BCE(y_pre, labels)
    return loss

class myGCN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        dataset,
        layer_type,
        NUM_HIDDEN_LAYERS=1,
        NUM_MLP_LAYERS=1,
        POST_NUM_MLP_LAYERS=1,
        aggr_fn="mean",
        dropout_rate=0.0,
        activation_fn_name="gelu",
        layer_norm_flag=False,
        model_name = "",
        **kwargs,
    ):
        super().__init__()

        torch.manual_seed(RANDOM_SHEET_NUMBER)
        self.NUM_HIDDEN_LAYERS = NUM_HIDDEN_LAYERS
        self.NUM_MLP_LAYERS = NUM_MLP_LAYERS
        self.POST_NUM_MLP_LAYERS = POST_NUM_MLP_LAYERS
        self.layer_norm_flag = layer_norm_flag
        self.dropout_rate = dropout_rate
        layer = get_layer_impl_from_layer_type[layer_type]["impl"]
        layer_params = get_layer_impl_from_layer_type[layer_type]["params"]
        layer_params['aggr'] = aggr_fn
        self.layer_norm = torch.nn.LayerNorm(hidden_channels)
        self.activation_fn = activation_name_implementation[activation_fn_name]()
        self.model_name = model_name
        layers = OrderedDict()
        for i in range(NUM_HIDDEN_LAYERS):
            print("added graphhidden layer: ", i)
            layers[str(i)] = layer(hidden_channels, hidden_channels, **layer_params)
            for k in layers[str(i)].state_dict().keys():
                torch.nn.init.xavier_uniform_(
                    layers[str(i)].state_dict()[k].reshape(1, -1)
                ).reshape(-1)
        self.hidden_layers = Sequential(layers)

        mlp_layers = OrderedDict()
        for i in range(NUM_MLP_LAYERS):
            print("added mlp layer: ", i)
            if i == 0:
                mlp_layers[str(i)] = torch.nn.Linear(
                    dataset.num_features, hidden_channels
                )
            else:
                mlp_layers[str(i)] = torch.nn.Linear(hidden_channels, hidden_channels)
            torch.nn.init.xavier_uniform_(mlp_layers[str(i)].weight)
        self.mlp_layers = Sequential(mlp_layers)

        post_mlp_layers = OrderedDict()
        for i in range(POST_NUM_MLP_LAYERS):
            print("added post-mlp layer: ", i)
            if i == 0:
                post_mlp_layers[str(i)] = Sequential(
                    torch.nn.LayerNorm(dataset.num_features+hidden_channels),
                    torch.nn.Linear(
                    dataset.num_features+hidden_channels, hidden_channels
                ))
            else:
                post_mlp_layers[str(i)] = Sequential(
                    torch.nn.LayerNorm(hidden_channels),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
            torch.nn.init.xavier_uniform_(post_mlp_layers[str(i)][1].weight)
        self.post_mlp_layers = Sequential(post_mlp_layers)
        self.lin1 = torch.nn.Linear(hidden_channels,dataset.num_classes)
        torch.nn.init.xavier_uniform_(self.lin1.weight)


    def forward(self, raw, edge_index, device, edge_weight=None):
        edge_index = edge_index.to(device)

        x = raw 
        for i in range(self.NUM_MLP_LAYERS):
            x = self.mlp_layers[i](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        node_embeddings = []
        for i in range(self.NUM_HIDDEN_LAYERS):  # 0
            if edge_weight is None:
                node_embeddings = self.hidden_layers[i](x, edge_index)
            else:
                node_embeddings = self.hidden_layers[i](x, edge_index, edge_weight)
            x = self.activation_fn(node_embeddings)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = torch.cat((raw, x), dim=1)
        for i in range(self.POST_NUM_MLP_LAYERS):
            x = self.post_mlp_layers[i](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin1(x)

        return x.sigmoid(), node_embeddings


def train(model, data, params_hparam, FOLDER_NAME, mode_training, pretrained_model_path=None, **kwargs):
    global DD_MM_YYYY
    global max_patience_count
    global max_num_epochs
    global last_loss
    global device
    global writer 
    
    max_num_epochs = params_hparam["NUM_EPOCHS"]
    max_epochs = max_num_epochs
    batch_size=params_hparam["batch_size"]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params_hparam["LR"], weight_decay=params_hparam["WD"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    epoch = 0
    DD_MM_YYYY_HH_MM_SS_epoch = datetime.now().strftime("%d%m%Y-%H%M%S")

    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path))


    train_nodes = data.train_mask.nonzero(as_tuple=False).squeeze()
    # Initialize NeighborSampler
    if mode_training == 'transductive':
        train_loader = NeighborSampler(data.edge_index, sizes=[params_hparam["NUM_NODES"]], batch_size=batch_size, shuffle=True)
    if mode_training == 'inductive':
        subgraph_edge_index, _ = subgraph(train_nodes, data.edge_index, relabel_nodes=True, )
        train_loader = NeighborSampler(subgraph_edge_index, sizes=[params_hparam["NUM_NODES"]], batch_size=batch_size, shuffle=True)
        
    while True:
        epoch += 1
        print("epoch: ", epoch)
        total_loss = 0

        model.train()
        for _, n_id, adjs in tqdm(train_loader):
            optimizer.zero_grad()
            y_hat, _ = model(data.x[n_id], adjs[0], device)
            train_loss =  get_loss(y_hat, data.y[n_id].float())
            del adjs
            torch.cuda.empty_cache()
            train_loss.backward()
            optimizer.step()
            total_loss += train_loss.item()
        lr_scheduler.step(train_loss)

        total_loss /= len(train_loader)
        writer.add_scalar('Loss/train', total_loss, epoch)
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x, data.edge_index, device)
            val_loss = get_loss(out[data.val_mask], data.y[data.val_mask].float())
            writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"Epoch: {epoch:03d}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss.item():.4f}")

        if val_loss < last_loss:
            epoch_fname = f"{FOLDER_NAME}/e{epoch}_valLos_{val_loss.item():.4f}.pt"
            saved[epoch_fname] = val_loss
            torch.save(model.state_dict(), f"{epoch_fname}")
            torch.save(model.state_dict(), f"{FOLDER_NAME}/best_model.pt")
            last_loss = val_loss
        elif val_loss >= last_loss:
            max_patience_count -= 1
            if max_patience_count == 0:
                break
        if epoch == max_epochs:
            break
    return optimizer, epoch, last_loss


def test(model, data, model_name, params_hparam):
    print("Testing...")
    model.eval()
    with torch.no_grad():
        out, node_embeddings = model(data.x, data.edge_index, device)
        out=out[data.test_mask]
        print("Test Loss: {:.4f}".format(get_loss(out, data.y[data.test_mask].float())))
        truth_labels = data.y[data.test_mask].cpu().detach().numpy()
        scores = out.cpu().detach().numpy()
        return scores, node_embeddings, truth_labels


def evaluation(scores, truth_labels):
    auc_c = metrics.roc_auc_score(truth_labels, scores, average=None)
    aucw = metrics.roc_auc_score(truth_labels, scores, average="weighted")
    auc_micro = metrics.roc_auc_score(truth_labels, scores, average="micro")
    auc_macro = metrics.roc_auc_score(truth_labels, scores, average="macro")
    writer.add_scalar("auc_micro", auc_micro)
    writer.add_scalar("auc_macro", auc_macro)
    writer.add_scalar("aucw", aucw)

    print(f"AUC: {auc_c}, {aucw}, {auc_micro}, {auc_macro}")
    return {
        "auc_c": auc_c,
        "aucw": aucw,
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
    }


def run_model(dataset, best_params, params_haparams, layer, FOLDER_NAME):
    global writer
    data = dataset[0]
    data.to(device)


    model = myGCN(**params_arch)
    print(model)
    optimizer, epoch, val_loss = train(model, data, params_hparam, FOLDER_NAME, mode_training=mode_training)
    best_model_fname = min(saved, key=saved.get)
    print("best model: ", best_model_fname)
    model.load_state_dict(torch.load(best_model_fname))
    scores, node_embeddings, truth_labels = test(model, data, layer, params_hparam)
    metrics_res = evaluation(scores, truth_labels)
    return model, metrics_res, node_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Run LECONV")
    parser.add_argument("--data_folder", type=str, required=True, help="Train/val/test PyG dataset folder")
    parser.add_argument("--model", type=str, default="LEConv")
    parser.add_argument("--activation", type=str, default="celu")
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--WD", type=float, default=0.00001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--NUM_NODES", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--NUM_HIDDEN_LAYERS", type=int, default=1)
    parser.add_argument("--POST_NUM_MLP_LAYERS", type=int, default=1)
    parser.add_argument("--NUM_MLP_LAYERS", type=int, default=1)
    parser.add_argument("--model_name", type=str, required=True, default="None")
    parser.add_argument("--model_folder", type=str, required=True, default="None")
    parser.add_argument("--mode_training", type=str, required=True, default="None", help="inductive or transductive")
    parser.add_argument("--outputdir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, required=True, default="")
    return parser.parse_args()


def save_tuple_of_string_to_file(filename, tuple_of_string):
    with open(filename, "w") as f:
        for item in tuple_of_string:
            f.write("%s" % item)

saved = {}
DD_MM_YYYY = datetime.now().strftime("%d_%m_%Y")
DATE_TIME = datetime.now().strftime("%d%m%Y-%H%M%S")
DAY_MONTH_YEAR = datetime.now().strftime("%d%m%Y")
RANDOM_SHEET_NUMBER = 42
last_loss = math.inf
max_patience_count = 15 # 10  # 0
max_epochs = 1

if __name__ == "__main__":
    global device
    global FOLDER    

    layer = parse_args().model
    model_name = parse_args().model_name
    model_folder = parse_args().model_folder
    mode_training = parse_args().mode_training
    args_dict = parse_args()
    lr = parse_args().lr
    NUM_NODES = parse_args().NUM_NODES
    hidden = parse_args().hidden
    dropout = parse_args().dropout
    WD = parse_args().WD
    batch_size = parse_args().batch_size
    outputdir=parse_args().outputdir
    NUM_HIDDEN_LAYERS=parse_args().NUM_HIDDEN_LAYERS
    POST_NUM_MLP_LAYERS=parse_args().POST_NUM_MLP_LAYERS
    NUM_MLP_LAYERS=parse_args().NUM_MLP_LAYERS
    activation = parse_args().activation
    aggr=parse_args().aggr
        
    if mode_training != 'transductive' and mode_training != 'inductive':
        raise Exception("mode_training must be either transductive or inductive but is: ", mode_training)

    MODEL_FOLDER = f"{model_folder}"
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    NUM_EPOCHS = parse_args().epochs
    experiment_name = parse_args().experiment_name
    data_folder = parse_args().data_folder
    cuda = parse_args().cuda
    device = torch.device(f"cuda:{str(cuda)}" if (torch.cuda.is_available() and cuda >= 0) else "cpu")
    
    print(f'Executing : {os.path.basename(__file__)}')
    FNAME = os.path.join(os.path.join(experiment_name, 'runs'), model_name)
    print('Tensorboard logs will be saved in: ', FNAME)
    writer = SummaryWriter(FNAME)
    
    print("device: ", device)
    print("DATA FOLDER: ", data_folder)
    print(">>> layer", layer)
    COMMIT_HASH = os.popen("git rev-parse HEAD").read()
    print("COMMIT_HASH: ", COMMIT_HASH)
    
    dataset = MyOwnDataset(data_folder)
    params_config = {
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "POST_NUM_MLP_LAYERS":POST_NUM_MLP_LAYERS,
        "NUM_MLP_LAYERS":NUM_MLP_LAYERS,
        "activation_fn_name":activation,
        "batch_size": batch_size,
        "WD": WD,
        "dropout_rate": dropout,
        "LR": lr,
        "hidden_channels": hidden,
        "NUM_NODES": NUM_NODES,
        "NUM_EPOCHS": NUM_EPOCHS,
        "layer": layer,
        "aggr": aggr
    }
    
    
    params_arch = {
        **{
            "hidden_channels": None,
            "dataset": dataset,
            "NUM_HIDDEN_LAYERS": None,
            "layer_type": layer,
            "aggr_fn": params_config['aggr'],  #
        },
        **params_config,
    }
    params_hparam = {**params_config}
    params_fpath = os.path.join(MODEL_FOLDER, "model_params.pt")
    print(f"<<<<< Model constructor params path: {params_fpath}")
    torch.save(params_arch, os.path.join(MODEL_FOLDER, "model_params.pt"))
    
    # >> params 
    # params_hparam = {
    #     **params_hparam,
    #     **{'hidden_channels': 2640, 'NUM_HIDDEN_LAYERS': 1, 'layer_type': 'SAGEConv', 'aggr_fn': 'min', 'POST_NUM_MLP_LAYERS': 3, 'NUM_MLP_LAYERS': 1, 'activation_fn_name': 'gelu', 'batch_size': 128, 'WD': 1e-05, 'dropout_rate': 0.4, 'LR': 0.0009067739710479194, 'NUM_NODES': 488, 'NUM_EPOCHS': 5, 'layer': 'SAGEConv', 'aggr': 'min'},
    #     **{'NUM_HIDDEN_LAYERS': 1, 'POST_NUM_MLP_LAYERS': 3, 'NUM_MLP_LAYERS': 2, 'LR': 0.000960432820518382, 'hidden_channels': 1608, 'NUM_NODES': 259, 'aggr': 'min'}
    # }
    all_models, all_metrics_res = [], []
    model, metrics_res, node_embeddings = run_model(dataset, params_config, params_hparam, layer, MODEL_FOLDER)
    to_save = {
        **{"gnn": layer},
        **{i: metrics_res["auc_c"][i] for i in range(len(metrics_res["auc_c"]))},
        **{
            "aucw": metrics_res["aucw"],
            "auc_micro": metrics_res["auc_micro"],
            "auc_macro": metrics_res["auc_macro"],
        },
    }

    df = pd.DataFrame(to_save, index=[0])
    print(df.T)

# python transductive.py --model SAGEConv --data_folder data_e_knn_nf_lstm_26_11_2022 --epochs 1000 # 0.76
# python transductive.py --model SAGEConv --data_folder data_e_knn_nf_lstm_26_11_2022 --epochs 10
# python train_GNN.py --model SAGEConv --data_folder data_e_random_nf_lstm_26_11_2022 --epochs 10
# python 28_11_22_train_GNN.py --model SAGEConv --data_folder data_e_random_nf_lstm_26_11_2022 --epochs 10
# python 28_11_22_train_GNN.py --model SAGEConv --data_folder expert_graph/A_expert_edges_exact --epochs 1000
# python 28_11_22_train_GNN.py --model LEConv --data_folder expert_graph/A_expert_edges_exact --epochs 1000

# python 28_11_22_train_GNN.py --model SAGEConv --data_folder expert_graph/A_expert_edges_exact
# python 28_11_22_train_GNN.py --model SAGEConv --data_folder expert_graph/A_m2_expert_edges_inter_category
# python 28_11_22_train_GNN.py --model SAGEConv --data_folder expert_graph/A_m3_expert_edges_inter_category
# python gnn_DSAA23.py --model SAGEConv --data_folder data_e_knn_nf_stat_26_11_2022 --epochs 1 --WD 0.001 --lr 0.0001 --hidden 8192 --batch_size 512 --model_name SAGEConv_nf_stat_es_knn_2011_05_19_13_55_26 --mode_training inductive
# python gnn_DSAA23.py --model SAGEConv --data_folder data_e_knn_nf_stat_26_11_2022 --epochs 1 --WD 0.001 --lr 0.0001 --hidden 8192 --batch_size 512 --model_name SAGEConv_nf_stat_es_knn_2011_05_19_13_55_26 --mode_training transductive
# python train_gnn.py --model SAGEConv --data_folder graphs/data_trivial_stat/processed/ --epochs 1 --WD 0.001 --lr 0.0001 --hidden 8192 --batch_size 512 --model_name SAGEConv_nf_stat_es_knn_2011_05_19_13_55_26 --mode_training transductive
    