from __future__ import absolute_import
from __future__ import print_function
from tqdm import tqdm
import os
import sys
from datetime import datetime
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import functools
import random

from torch.utils.tensorboard import SummaryWriter

path_module = os.path.join('.','..')
print(path_module)
sys.path.append('../..') ## Needed for mimic3models
sys.path.append('../../..')
from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import common_utils
from mimic3models.lstm import LSTM_PT
from mimic3models.losses import SupConLoss_MultiLabel, SupNCELoss, CBCE_loss, CBCE_WithLogitsLoss
from mimic3models.torch_utils import Dataset, optimizer_to, model_summary, TimeDistributed, shuffle_within_labels,shuffle_time_dim
from mimic3models.time_report import TimeReport

print = functools.partial(print, flush=True)
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
timestamp_only_day = datetime.now().strftime('%Y-%m-%d')
print('Timestamp: {}'.format(timestamp))

# Arguments:
parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0) 
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task', required=True)
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',                     default=f'./pytorch_states_{timestamp}/BCE/')
# New added
parser.add_argument('--seed', type=int, default=1, help='Random seed manually for reproducibility.')
parser.add_argument('--hidden_folder', type=str, help='Path to the data of phenotyping task', default = f'data/BCE_LSTM_{timestamp_only_day}/')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--hidden_linear_dim', type=int, default=256)
parser.add_argument('--num_linear_layers', type=int, default=2)
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--coef_contra_loss', type=float, default=0, help='CE + coef * contrastive loss')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay of adam')
parser.add_argument('--final_act', type=str, default='sigmoid', help='sigmoid or softmax')
args = parser.parse_args()
# print(args)
# for arg in vars(args):
#     print(arg, getattr(args, arg))
# print hidden_folder
if not os.path.exists(args.hidden_folder):
    os.makedirs(args.hidden_folder)
    print(f'Created hidden folder: {args.hidden_folder}')

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)  # cpu
torch.cuda.manual_seed(args.seed)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')
train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_listfile.csv'))

val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                               listfile=os.path.join(args.data, 'val_listfile.csv'))

test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                listfile=os.path.join(args.data, 'test_listfile.csv'))

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ph_ts{}.input_str-previous.start_time-zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.join(os.path.dirname(__file__), 'phenotyping'), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['num_classes'] = 25
args_dict['target_repl'] = target_repl

# GPU, Model, Optimizer, Criterion Setup/Loading
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: Got {} CUDA devices! Probably run with --cuda".format(torch.cuda.device_count()))

#cuda is available
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
print('Using device: ', device)
subfolder = timestamp + '_'.join([str(x) for x in (76, args.dim, args.depth, 25, args.dropout)])

MNAME = f'runs_{timestamp_only_day}/runs_{subfolder}'
writer = SummaryWriter(MNAME)

# # Build the model
# if args.network == "lstm":
model = LSTM_PT(
    input_dim=76, 
    batch_size=args.batch_size,
    hidden_dim=args.dim, 
    hidden_linear_dim=args.dim, 
    num_layers=args.depth, 
    num_linear_layers=2,
    num_classes=25,
    dropout=args.dropout, 
    target_repl=False, 
    deep_supervision=False, 
    # final_act=nn.Sigmoid,#args.final_act,
    task='ph'
)

if target_repl:
    raise NotImplementedError
else:
    criterion_BCE = nn.BCELoss()
    criterion_SCL_MultiLabel = SupConLoss_MultiLabel(temperature=0.1)   # temperature=0.01)  # temperature=opt.temp

    def get_loss(y_pre, labels, representation, alpha=0):
        # CBCE_WithLogitsLoss is more numerically stable than CBCE_Loss when model is complex/overfitting
        if y_pre.shape != labels.shape:
            y_pre=y_pre.resize(*labels.shape)
        loss = criterion_BCE(y_pre, labels)
        if alpha > 0:
            if len(representation.shape) == 2:
                representation = representation.unsqueeze(1)
            scl_loss = criterion_SCL_MultiLabel(representation, labels)
            loss = loss + alpha * scl_loss
        return loss

# set lr and weight_decay later # or use other optimization say adamW later?
optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)
training_losses = []
validation_losses = []
test_losses = []
validation_results = []
test_results = []
model_names = []

# Load model weights # n_trained_chunks = 0
start_from_epoch = 0
if args.load_state != "":
    print('Load model state from: ', args.load_state)
    checkpoint = torch.load(args.load_state, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_from_epoch = checkpoint['epoch']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    validation_results = checkpoint['validation_results']
    test_results = checkpoint['test_results']
    optimizer_to(optimizer, device)
    print("Load epoch: ", start_from_epoch, 'Load model: ', )
    print('Load model done!')

model.to(device)
try:
    criterion_SCL_MultiLabel.to(device)
except NameError:
    print("No criterion_SCL_MultiLabel")
print(model)
model_summary(model)

# Training & Testing parts:
if args.mode == 'train':
    print('Training part: beginning loading training & validation datasets...')
    start_time = time.time()

    train_data_gen = utils.BatchGen(train_reader, discretizer, normalizer, args.batch_size,
                                    args.small_part, target_repl, shuffle=True, return_names=True)
    val_data_gen = utils.BatchGen(val_reader, discretizer, normalizer, args.batch_size,
                                  args.small_part, target_repl, shuffle=False, return_names=True)
    test_data_gen = utils.BatchGen(test_reader, discretizer, normalizer, args.batch_size,
                                   args.small_part, 
                                   target_repl=target_repl, shuffle=False, return_names=True)

    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Load data done. Elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))
    print('Data summary:')
    print('len(train_data_gen):', len(train_data_gen.names),
          'len(val_data_gen):', len(val_data_gen.names),
          'len(test_data_gen):', len(test_data_gen.names))
    print('batch size:', args.batch_size, 'epoch:', args.epochs, 'iters/epoch:', train_data_gen.steps)

    print("Beginning model training...")
    iter_per_epoch = train_data_gen.steps
    tr = TimeReport(total_iter=args.epochs * train_data_gen.steps)
    
    LOG_LOSSES = []
    LOG_STEPS =[]
    for epoch in tqdm(range(1+start_from_epoch, 1+args.epochs)): #tqdm
        model.train()
        train_losses_batch = []
        EPOCH_NAME = str(epoch)
        for i in tqdm(range(train_data_gen.steps)):  # tqdm
            
            BATCH_NAME = str(i)
            ret = next(train_data_gen)
            X_batch_train, labels_batch_train, x_length = ret["data"]
            name_batch_train = ret["names"]

            X_batch_train = torch.tensor(X_batch_train, dtype=torch.float32)
            labels_batch_train = torch.tensor(labels_batch_train, dtype=torch.float32)
            X_batch_train = rnn_utils.pack_padded_sequence(X_batch_train, x_length, batch_first=True)

            optimizer.zero_grad()

            X_batch_train = X_batch_train.to(device)
            labels_batch_train = labels_batch_train.to(device)
            bsz = labels_batch_train.shape[0]

            y_hat_train, y_representation, hn, cn = model(X_batch_train)
            if y_hat_train.ndim == 1:
                    y_hat_train=y_hat_train.unsqueeze(dim=0)
            loss = get_loss(y_hat_train, labels_batch_train, y_representation, args.coef_contra_loss)
            LOG_LOSSES.append(loss.item())
            LOG_STEPS.append(EPOCH_NAME+BATCH_NAME)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)  # args.clip) seems little effect
            optimizer.step()
            train_losses_batch.append(loss.item())
            tr.update()

        training_loss = np.mean(train_losses_batch)
        training_losses.append(training_loss)

        # Validation Part
        print('Validation results:')
        with torch.no_grad():
            model.eval()
            val_losses_batch = []
            predicted_prob_val = []
            true_labels_val = []
            for i in range(val_data_gen.steps):
                ret = next(val_data_gen)
                X_batch_val, labels_batch_val, x_length = ret["data"]
                name_batch_val = ret["names"]

                X_batch_val = torch.tensor(X_batch_val, dtype=torch.float32)
                labels_batch_val = torch.tensor(labels_batch_val, dtype=torch.float32)
                X_batch_val = rnn_utils.pack_padded_sequence(X_batch_val, x_length, batch_first=True)

                X_batch_val = X_batch_val.to(device)
                labels_batch_val = labels_batch_val.to(device)
                bsz = labels_batch_val.shape[0]

                y_hat_val, y_representation_val, hn, cn = model(X_batch_val)
                val_loss_batch = get_loss(y_hat_val, labels_batch_val, y_representation_val, args.coef_contra_loss)
                if y_hat_val.ndim == 1:
                    y_hat_val=y_hat_val.unsqueeze(dim=0)
                val_losses_batch.append(val_loss_batch.item())
                # predicted labels
                predicted_prob_val.append(y_hat_val)
                true_labels_val.append(labels_batch_val)

            validation_loss = np.mean(val_losses_batch)
            validation_losses.append(validation_loss)

            predicted_prob_val = torch.cat(predicted_prob_val, dim=0).cpu().detach().numpy()
            true_labels_val = torch.cat(true_labels_val, dim=0).cpu().detach().numpy()
            val_result = metrics.print_metrics_multilabel(true_labels_val, predicted_prob_val, verbose=0)
            print(val_result)
            validation_results.append(val_result)

        # Additional test part. God View. should not used for model selection
        print('Test results:')
        with torch.no_grad():
            model.eval()
            predicted_prob_test = []
            true_labels_test = []
            name_test = []
            test_losses_batch = []
            for i in range(test_data_gen.steps):
                ret = next(test_data_gen)
                X_batch_test, y_batch_test, x_length = ret["data"]
                name_batch_test = ret["names"]

                X_batch_test = torch.tensor(X_batch_test, dtype=torch.float32)
                y_batch_test = torch.tensor(y_batch_test, dtype=torch.float32)
                X_batch_test = rnn_utils.pack_padded_sequence(X_batch_test, x_length, batch_first=True)

                X_batch_test = X_batch_test.to(device)
                y_batch_test = y_batch_test.to(device)
                bsz = y_batch_test.shape[0]

                y_hat_batch_test, y_representation_test, hn, cn = model(X_batch_test)
                test_loss_batch = get_loss(y_hat_batch_test, y_batch_test, y_representation_test, args.coef_contra_loss)
                test_losses_batch.append(test_loss_batch.item())

                if y_hat_batch_test.ndim == 1:
                    y_hat_batch_test=y_hat_batch_test.unsqueeze(dim=0)
                predicted_prob_test.append(y_hat_batch_test)
                true_labels_test.append(y_batch_test)
                name_test.append(name_batch_test)

            test_loss = np.mean(test_losses_batch)
            test_losses.append(test_loss)

            predicted_prob_test = torch.cat(predicted_prob_test, dim=0)
            true_labels_test = torch.cat(true_labels_test, dim=0)  # with threshold 0.5, not used here
            predictions_test = (predicted_prob_test.cpu().detach().numpy())
            true_labels_test = true_labels_test.cpu().detach().numpy()
            name_test = np.concatenate(name_test)
            test_result = metrics.print_metrics_multilabel(true_labels_test, predictions_test, verbose=1)

            print(test_result)
            test_results.append(test_result)
        print('Epoch [{}/{}], {} Iters/Epoch, training_loss: {:.3f}, validation_loss: {:.3f}, test_loss: {:.3f},'
              '{:.2f} sec/iter, {:.2f} iters/sec: '.
              format(epoch, args.epochs, iter_per_epoch,
                     training_loss, validation_loss, test_loss,
                     tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
       
        tr.print_summary()
        print("=" * 50)

        model_final_name = model.say_name()
        path = os.path.join(args.output_dir + model_final_name
                            + '.BCE+SCL.a{}.bs{}.wdcy{}.epo{}.'
                              'Val-AucMac{:.4f}.AucMic{:.4f}.'
                              'Tst-AucMac{:.4f}.AucMic{:.4f}'.
                            format(args.coef_contra_loss, args.batch_size, args.weight_decay, epoch,
                                   val_result['ave_auc_macro'], val_result['ave_auc_micro'],
                                   test_result['ave_auc_macro'], test_result['ave_auc_micro']))

        model_names.append(path + '.pt')
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if args.save_every and epoch % args.save_every == 0:
            # Set model checkpoint/saving path
            test_details = {
                'name': name_test,
                'prediction': predictions_test,
                'true_labels': true_labels_test
            }
            torch.save({
                'model_str': model.__str__(),
                'args:': args,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'validation_results': validation_results,
                'test_results': test_results,
                'test_details': test_details,
            }, path+'.pt')
            print('\n-----Save model: \n{}\n'.format(path+'.pt'))

            # pd_test = pd.DataFrame(data=test_details)  # , index=range(1, len(validation_results) + 1))
            # pd_test.to_csv(path + '_[TEST].csv')
        
        writer.add_scalar('Loss/train',training_loss, epoch)
        writer.add_scalar('Loss/val',validation_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train/val/macro ',val_result['ave_auc_macro'], epoch)
        # writer.add_scalar('Accuracy/test', np.random.random(), epoch)
    print('Training complete...')
    
    best_epoch = np.argmax([x['ave_auc_macro'] for x in validation_results])
    print('Best epoch: {}'.format(best_epoch))

    # Load the model with the best epoch
    model_final_name = model.say_name()
    path = os.path.join(args.output_dir + model_final_name
                        + '.BCE+SCL.a{}.bs{}.wdcy{}.epo{}.'
                        'Val-AucMac{:.4f}.AucMic{:.4f}.'
                        'Tst-AucMac{:.4f}.AucMic{:.4f}'.
                        format(args.coef_contra_loss, args.batch_size, args.weight_decay, best_epoch+1,
                            validation_results[best_epoch]['ave_auc_macro'], validation_results[best_epoch]['ave_auc_micro'],
                            test_results[best_epoch]['ave_auc_macro'], test_results[best_epoch]['ave_auc_micro']))
    checkpoint = torch.load(path+'.pt')

    # Get the predicted scores and true labels from the test set
    test_details = checkpoint['test_details']
    name_test = test_details['name']
    true_labels_test = test_details['true_labels']
    predicted_prob_test = test_details['prediction']

    # Convert the data to a dataframe
    df = pd.DataFrame({'Name': name_test, 'True Labels': true_labels_test.tolist(), 'Predicted Scores': predicted_prob_test.tolist()})
    df.to_csv('predictions.csv', index=False)
    print('Predictions saved to predictions.csv')
    
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Total Elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))

    r = {
        'ave_auc_macro-val': [x['ave_auc_macro'] for x in validation_results],
        'ave_auc_macro-test': [x['ave_auc_macro'] for x in test_results],
        'ave_auc_micro-val': [x['ave_auc_micro'] for x in validation_results],
        'ave_auc_micro-test': [x['ave_auc_micro'] for x in test_results],
    }
    pdr = pd.DataFrame(data=r, index=range(1, len(validation_results)+1))
    ax = pdr.plot.line()
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':') #, linewidth='0.5', color='black')
    plt.grid()
    fig = ax.get_figure()
    fig.savefig('results.png')
    r_all = {
        'model-name': model_names,
        'ave_auc_macro-val': [x['ave_auc_macro'] for x in validation_results],
        'ave_auc_micro-val': [x['ave_auc_micro'] for x in validation_results],
        'ave_auc_weighted-val': [x['ave_auc_weighted'] for x in validation_results],
        'ave_auc_macro-test': [x['ave_auc_macro'] for x in test_results],
        'ave_auc_micro-test': [x['ave_auc_micro'] for x in test_results],
        'ave_auc_weighted-test': [x['ave_auc_weighted'] for x in test_results],
    }
    
    pd_r_all = pd.DataFrame(data=r_all, index=range(1, len(validation_results) + 1))
    pd_r_all.to_csv('results.csv')
    print('Dump', path + '[.png/.csv] done!')
elif args.mode == 'test':
    print('Beginning testing...')
    start_time = time.time()
    # ensure that the code uses test_reader
    model.to(torch.device('cpu'))

    del train_reader
    del val_reader

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

    test_data_gen = utils.BatchGen(test_reader, discretizer,
                                   normalizer, args.batch_size,
                                   args.small_part, target_repl,
                                   shuffle=False, return_names=True)
    names = []
    ts = []
    labels = []
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(test_data_gen.steps)):
            print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
            ret = next(test_data_gen)
            x = ret["data"][0]
            y = ret["data"][1]
            x_length = ret["data"][2]
            cur_names = ret["names"]
            cur_ts = ret["ts"]
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            x_pack = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)

            pred, representation, hn, cn = model(x_pack)
            predictions.append(pred)
            labels.append(y)
            names += list(cur_names)
            ts += list(cur_ts)

        predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
        labels = torch.cat(labels, dim=0).cpu().detach().numpy()

    results = metrics.print_metrics_multilabel(labels, predictions)
    print(results)
    print('Format print :.4f for results:')
    print(test_results)
    # TODO
    # if boostrap:
    #     from utils import boostrap_interval_and_std
    #     pd_bst = boostrap_interval_and_std(predictions, true_labels, 100)
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option("precision", 4)
    #     print(pd_bst.describe())

    path = os.path.join("test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, ts, predictions, labels, path)
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Testing elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))

elif args.mode == 'save':
    sets = ['train', 'val', 'test']
    for set in sets:
        print('Processing {} set...'.format(set))
        start_time = time.time()
        model.to(torch.device('cpu'))

        if set == 'train':
            reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                listfile=os.path.join(args.data, 'train_listfile.csv'))
        elif set == 'val':
            reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                            listfile=os.path.join(args.data, 'val_listfile.csv'))
        else:
            reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

        data_gen = utils.BatchGen(reader, discretizer, normalizer, args.batch_size,
                                args.small_part, target_repl, shuffle=False, return_names=True)
        names = []
        ts = []
        labels = []
        predictions = []
        hiddens_hn = []  # Store hn
        hiddens_cn = []  # Store cn
        with torch.no_grad():
            for i in tqdm(range(data_gen.steps)):
                print("Processing {} / {} in {} set".format(i, data_gen.steps, set), end='\r')
                ret = next(data_gen)
                x = ret["data"][0]
                y = ret["data"][1]
                x_length = ret["data"][2]
                cur_names = ret["names"]
                cur_ts = ret["ts"]
                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                x_pack = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)

                pred, representation, hn, cn = model(x_pack)
                predictions.append(pred)
                labels.append(y)
                names += list(cur_names)
                ts += list(cur_ts)
                hiddens_hn.append(hn)  # Append hn
                hiddens_cn.append(cn)  # Append cn

            predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
            labels = torch.cat(labels, dim=0).cpu().detach().numpy()
            hiddens_hn_reshaped = [hidden_hn.reshape(-1, 256).detach().cpu().numpy() for hidden_hn in hiddens_hn]
            hiddens_hn_concat = np.concatenate(hiddens_hn_reshaped, axis=0)
        hidden_folder = args.hidden_folder

        # Save as a .npy file
        results = metrics.print_metrics_multilabel(labels, predictions)
        print(results)
        
        # Save hidden states to a specified directory
        print('Saving hidden states to {}'.format(hidden_folder))
        os.makedirs(hidden_folder, exist_ok=True)
        # saved in 
        print('Saving hidden states to {}'.format(hidden_folder))
        np.save(os.path.join(args.hidden_folder, f"hidden_hn_{set}.npy"), hiddens_hn_concat)
        np.save(os.path.join(hidden_folder, f'labels_{set}.npy'), labels)
        np.save(os.path.join(hidden_folder, f'predictions_{set}.npy'), predictions)
        np.save(os.path.join(hidden_folder, f'name_{set}.npy'), names)

        path = os.path.join(f"{set}_predictions", os.path.basename(args.load_state)) + ".csv"
        utils.save_results(names, ts, predictions, labels, path)
        h_, m_, s_ = TimeReport._hms(time.time() - start_time)
        print('{} set processing elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(set.capitalize(), h_, m_, s_))
else:
    raise ValueError("Wrong value for args.mode")
