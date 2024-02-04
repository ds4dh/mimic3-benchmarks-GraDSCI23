import math
from torch import Tensor
from torch.nn.parameter import Parameter
import numpy as np
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
# from mimic3models.pytorch_models.torch_utils import TimeDistributed#, LastTimestep
# from mimic3models.pytorch_models.MGL import MGL
from mimic3models.torch_utils import TimeDistributed#, LastTimestep
from mimic3models.MGL import MGL # GNU licence from https://github.com/yaquanzhang/mGRN for early experiments

# from pytorch_model_summary import summary
# from wrappers import TimeDistributed



# if cuda is available, use it
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available on this device...")
def predict_labels(input):
    """
    Args:
        input: N_samples * C_classes. The input is expected to contain raw, unnormalized scores for each class.
        The input is the same as the input of  torch.nn.CrossEntropyLoss
    Outputs:
        p: N_samples * C_classes, the normalized probability
        labels: N_samples * 1, the class labels
    """
    p = F.softmax(input, dim=1)
    labels = torch.argmax(p, dim=1)
    return p, labels


def squash_packed_iid(x, fn):
    """
    Applying fn to each element of x i.i.i
    x is torch.nn.utils.rnn.PackedSequence
    """
    return PackedSequence(fn(x.data), x.batch_sizes,
                          x.sorted_indices, x.unsorted_indices)


class LSTM_PT(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_linear_dim, num_layers=2, num_linear_layers=2, time_distributed_wrapper_flag=True,discount_linear_dim=True, 
                 num_classes=2,
                 dropout=0.3, target_repl=False, batch_size = 64,deep_supervision=False, task='ihm', final_act=None, **kwargs):
        super(LSTM_PT, self).__init__()

        print("==> not used params in network class:", kwargs.keys())
        self.time_distributed_wrapper_flag=time_distributed_wrapper_flag
        self.discount_linear_dim=discount_linear_dim
        self.input_dim = input_dim  # 76
        self.hidden_dim = hidden_dim  # 16
        self.hidden_linear_dim = hidden_linear_dim
        self.num_linear_layers = num_linear_layers  # using 2 here
        self.num_layers = num_layers  # using 2 here
        self.num_classes = num_classes  # 2, for binary classification using (softmax + ) cross entropy
        self.dropout = dropout  # 0.3
        self.batch_size = batch_size
        if final_act is None:
            # Set default activation
            if task in ['decomp', 'ihm', 'ph']:
                self.final_activation = nn.Sigmoid()
            elif task in ['los']:
                if num_classes == 1:
                    self.final_activation = nn.ReLU()
                else:
                    self.final_activation = nn.Softmax()
            else:
                raise ValueError("Wrong value for task")
        else:
            self.final_activation = final_act

        # Configurations
        self.bidirectional = True  # default bidirectional for the LSTM layer except output LSTM layer
        if deep_supervision:
            self.bidirectional = False

        # Main part of the network - pytorch
        self.input_dropout_layer = nn.Dropout(p=self.dropout)

        num_hidden_dim = self.hidden_dim
        if self.bidirectional:
            num_hidden_dim = num_hidden_dim // 2
        self.MGL = MGL(
            in_features=self.input_dim, num_moments=10, hidden_features=num_hidden_dim, out_features=100)
        if self.num_layers > 1:
            self.main_lstm_layer = nn.LSTM(
                input_size=self.input_dim,  # 76
                hidden_size=num_hidden_dim,  # 8
                batch_first=True,  # X should be:  (batch, seq, feature)
                dropout=self.dropout,
                # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
                num_layers=self.num_layers - 1,
                bidirectional=self.bidirectional)
            num_input_dim = self.hidden_dim  # for output lstm layer
        else:
            num_input_dim = self.input_dim  # for output lstm layer

        self.inner_dropout_layer = nn.Dropout(p=self.dropout)
        # Output module of the network - Pytorch
        # output lstm is not bidirectional. So, what if num_layers = 1, then, is_bidirectional is useless.
        # return_sequences = (target_repl or deep_supervision) # always return sequence in pytorch
        # Should I add input dropout layer here?
        self.output_lstm_layer = nn.LSTM(
            input_size=num_input_dim,  # output 1 layer
            hidden_size=self.hidden_dim,
            batch_first=True,  # X should be:  (batch, seq, feature)
            dropout=self.dropout, # only one layer, no need for inner dropout
            num_layers=1,  # output 1 layer
            bidirectional=False)  # output one direction for the output layer

        self.output_dropout_layer = nn.Dropout(p=self.dropout)

        if True:
        # if target_repl:
            
            # y = TimeDistributed(Dense(num_classes, activation=self.final_activation),
            #                     name='seq')(L)
            
            self.output_linear = TimeDistributed(nn.Linear(self.hidden_dim, self.num_classes),
                                # name='seq'
                                ) if self.time_distributed_wrapper_flag else nn.Linear(self.hidden_dim, self.num_classes)
            # y_last = LastTimestep(name='single')(y)
            # outputs = [y_last, y]
            # raise NotImplementedError``
        if deep_supervision:
            # y = TimeDistributed(Dense(num_classes, activation=self.final_activation))(L)
            # y = ExtendMask()([y, M])  # this way we extend mask of y to M
            # outputs = [y]
            raise NotImplementedError
        # else:
            # Only use last output.
            # y = Dense(num_classes, activation=final_activation)(L)
            # outputs = [y]
        self.output_linear2 = nn.Linear(self.hidden_dim, self.num_classes)  # , bias=False
        seq =[]
        for i in range(self.num_linear_layers):
            dimensions_in = int(self.hidden_linear_dim/(i+1)) if self.discount_linear_dim else self.hidden_linear_dim
            dimensions_out = int(self.hidden_linear_dim/(i+2)) if self.discount_linear_dim else self.hidden_linear_dim
            seq.append(
                TimeDistributed(nn.Linear(dimensions_in, dimensions_out),) 
                if self.time_distributed_wrapper_flag else  nn.Linear(dimensions_in, dimensions_out))
            seq.append(nn.CELU(inplace=True))
            seq.append(nn.Dropout(p=self.dropout))
            
        self.head = nn.Sequential(
            *seq                
            # TimeDistributed(nn.Linear(int(self.hidden_dim/1.5), int(self.hidden_dim/2.5)),),
            # nn.CELU(inplace=True),
            # TimeDistributed(nn.Linear(int(self.hidden_dim/2.5), int(self.hidden_dim/3.5)),),
            # nn.CELU(inplace=True),
        )

        self.reset_parameters()
        # taking care of initialization problem later
        # self.hidden = (
        #     torch.randn(self.num_layers * (2 ** int(self.bidirectional)), self.batch_size, self.hidden_dim[0]).to(
        #         self.device),
        #     torch.randn(self.num_layers * (2 ** int(self.bidirectional)), self.batch_size, self.hidden_dim[0]).to(
        #         self.device)
        # )

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
    
    def forward(self, raw_x_input):
        if isinstance(raw_x_input, PackedSequence):
            x1 = squash_packed_iid(raw_x_input, 
                                #   self.input_dropout_layer
                                lambda x:x
                                )
        else:
            x1 = self.input_dropout_layer(raw_x_input)  # (bs, seq=48, dim=76)

        # self.MGL(x1)
        if self.num_layers > 1:
            x3, (hn, cn) = self.main_lstm_layer(
                x1)  
            # (bs, seq=48, dim=16), hn, cn (num_layers * num_directions=2, batch, hidden_size = 8) add h0, c0 in the initilization
            if isinstance(x3, PackedSequence):
                x4 = squash_packed_iid(x3, self.inner_dropout_layer)
            else:
                x4 = self.inner_dropout_layer(x3)  # (bs, seq=48, dim=16)
            x2, (hn, cn) = self.output_lstm_layer(x4)  # (bs, seq=48, hidden_dim=16), hn,cn (1, bs, hidden_dim=16)
        else:
            x2, (hn, cn) = self.output_lstm_layer(x1)  # initilization

        #
        if isinstance(x2, PackedSequence):
            x5, lens_unpacked = pad_packed_sequence(x2, batch_first=True)
            indices = lens_unpacked - 1
            last_time_step = torch.gather(x5, 1,indices.view(-1, 1).unsqueeze(2).repeat(1, 1, x5.shape[2]).to(device=x5.device))
            
            last_time_step = last_time_step.squeeze()
        else:
            last_time_step = x2[:, -1, :]  # (bs, hidden_dim=16) lstm_out[:,-1,:] for batch first or h_n[-1,:,:]

        # last_time_step = self.output_dropout_layer(last_time_step)
        # try:
        # representation = F.normalize(last_time_step, dim=1)
        representation = last_time_step #F.normalize(last_time_step, dim=1)
        if torch.isnan(representation).any():
            import pdb; pdb.set_trace()
        # except:
        # import pdb; pdb.set_trace()
        # representation = F.normalize(self.head(last_time_step), dim=1)
        # representation = last_time_step  # for tSNE plot. should I move the normailization to loss part?
        # out = self.output_linear(representation)
        out = self.output_linear(last_time_step)  # (bs, 2)
        # out=self.output_dropout_layer2(out)
        # out= self.output_linear2(out)
        # out = self.head(last_time_step)
        # representation = F.normalize(out, dim=1)
        # No softmax activation if for pytorch crossentropy loss
        # original used in keras. should use sigmoid activation before keras binary cross entropy loss
        # import pdb; pdb.set_trace()
        out = self.final_activation(out).squeeze()
        return out, representation, hn, cn

    def say_name(self):
        return "{}.i{}.h{}.L{}.c{}{}".format('LSTM',
                                             self.input_dim,
                                             self.hidden_dim,
                                             self.num_layers,
                                             self.num_classes,
                                             ".D{}".format(self.dropout) if self.dropout > 0 else "-"
                                             )


class Linear_norm(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(Linear_norm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, F.normalize(self.weight, dim=1), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight.data)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.xavier_uniform_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)


def predict_model(model, data, batch_size, device):
    print('Starting predictions...')
    data_loader = DataLoader(dataset=data, batch_size=batch_size, drop_last=True)
    y_hat = torch.empty(data_loader.batch_size, 1).to(device)

    with torch.no_grad():
        for X_batch in data_loader:
            y_hat_batch = model(X_batch)
            y_hat = torch.cat([y_hat, y_hat_batch])

    y_hat = torch.flatten(y_hat[batch_size:, :]).cpu().numpy()  # y_hat[batchsize:] is to remove first empty 'section'
    print('Predictions complete...')
    return y_hat


def train_model(model, train_data, train_labels, test_data, test_labels, batch_size, num_epochs, device):
    model.apply(initialize_weights)

    training_losses = []
    validation_losses = []

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    train_hist = np.zeros(num_epochs)

    X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_labels, train_size=0.8)

    train_dataset = TensorDataset(X_train, y_train)
    validation_dataset = TensorDataset(X_validation, y_validation)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model.train()

    print("Beginning model training...")

    for t in range(num_epochs):
        train_losses_batch = []
        for X_batch_train, y_batch_train in train_loader:
            y_hat_train = model(X_batch_train)
            loss = loss_function(y_hat_train.float(), y_batch_train)
            train_loss_batch = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses_batch.append(train_loss_batch)

        training_loss = np.mean(train_losses_batch)
        training_losses.append(training_loss)

        with torch.no_grad():
            val_losses_batch = []
            for X_val_batch, y_val_batch in val_loader:
                model.eval()
                y_hat_val = model(X_val_batch)
                val_loss_batch = loss_function(y_hat_val.float(), y_val_batch).item()
                val_losses_batch.append(val_loss_batch)
            validation_loss = np.mean(val_losses_batch)
            validation_losses.append(validation_loss)

        print(f"[{t + 1}] Training loss: {training_loss} \t Validation loss: {validation_loss} ")

    print('Training complete...')
    return model.eval()


if __name__ == "__main__":
    torch.manual_seed(42)

    bs = 3
    seq = 48
    input_dim = 76
    num_layers = 2
    hidden_dim = 16
    print('Starting LSTM test...')
    X = torch.randn(bs, seq, input_dim)  # batch, seq_len,  input_size
    h0 = torch.randn(2 * num_layers, bs, hidden_dim)  # num_layers * num_directions, batch, hidden_size
    c0 = torch.randn(2 * num_layers, bs, hidden_dim)  # num_layers * num_directions, batch, hidden_size

    mylstm = LSTM_PT(input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=2,
                     dropout=0.3, target_repl=False, deep_supervision=False)
    # output: (seq_len, batch, num_directions * hidden_size)
    Y = mylstm(X)  # , (h0, c0)
    labels = predict_labels(Y)

    