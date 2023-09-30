# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd

import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse
import pickle

# Import model
from model import RNN
# Import functions
from utils import dataset_preparation, make_noise


# Goggle
from goggle.GoggleModel import GoggleModel

# Synthcity
from synthcity.plugins.core.dataloader import GenericDataLoader

torch.set_printoptions(edgeitems=3, precision=4, linewidth=200)

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
if not os.path.isdir('logs'):
    os.makedirs('logs')
log_file = 'logs/log_{}.log'.format(datetime.datetime.now().strftime("%Y_%B_%d_%I-%M-%S%p"))
open(log_file, 'a').close()

# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log(str): logger.info(str)


parser = argparse.ArgumentParser(description="DomainGen_Graph")

datasets = ['ONP', 'Moons', 'MNIST', 'Elec2', 'Shuttle']
parser.add_argument("--dataset", default="Elec2", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))

# Hyper-parameters
parser.add_argument("--noise_dim", default=16, type=int,
                    help="the dimension of the LSTM input noise.")
parser.add_argument("--num_rnn_layer", default=16, type=int,
                    help="the number of RNN hierarchical layers.")
parser.add_argument("--latent_dim", default=8, type=int,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--hidden_dim", default=16, type=int,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--noise_type", choices=["Gaussian", "Uniform"], default="Gaussian",
                    help="The noise type to feed into the generator.")

parser.add_argument("--num_workers", default=0, type=int,
                    help="the number of threads for loading data.")
parser.add_argument("--epoches", default=20, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--batch_size", default=16, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--learning_rate", default=1e-4, type=float,
                    help="the unified learning rate for each single task.")
parser.add_argument("--dropout_rate", default=0.5, type=float,
                    help="the dropout rate or LSTM.")

parser.add_argument("--is_test", default=True, type=bool,
                    help="if this is a testing period.")

parser.add_argument("--gpu_id", default=0, type=int,
                    help="the gpu id of for the exp.")

parser.add_argument("--bce_w", default=0.0, type=float,
                    help="the weight of BCE.")

args = parser.parse_args()


log('Is GPU available? {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
    device = f"cuda:{args.gpu_id}"
else:
    device = "cpu"
print(f"Current device = {device}")
print(f"=== dataset: {args.dataset}, num_rnn_layer: {args.num_rnn_layer}, latent_dim: {args.latent_dim}, hidden_dim: {args.hidden_dim} lr: {args.learning_rate}, bce_w: {args.bce_w}, epoches: {args.epoches} ===")

save_dir = f"data/{args.dataset}"
path_true_list_train_graph = f"{args.dataset}_true_list_train_graph_lr-0.01.pkl"
path_true_list_gt_graph = f"{args.dataset}_true_list_gt_graph_lr-0.01.pkl"
path_true_list_val_graph = f"{args.dataset}_true_list_val_graph_lr-0.01.pkl"
path_true_list_val_gt_graph = f"{args.dataset}_true_list_val_gt_graph_lr-0.01.pkl"

with open(os.path.join(save_dir, path_true_list_train_graph), 'rb') as f:
    true_list_train_graph = pickle.load(f)

with open(os.path.join(save_dir, path_true_list_gt_graph), 'rb') as f:
    true_list_gt_graph = pickle.load(f)

with open(os.path.join(save_dir, path_true_list_val_graph), 'rb') as f:
    true_list_val_graph = pickle.load(f)

with open(os.path.join(save_dir, path_true_list_val_gt_graph), 'rb') as f:
    true_list_val_gt_graph = pickle.load(f)


def fill_diagonal_with_ones(matrix):
    rows, cols = matrix.shape
    row_indices = torch.arange(0, rows).long()
    col_indices = torch.arange(0, cols).long()
    if matrix.is_cuda:
        row_indices = row_indices.cuda()
        col_indices = col_indices.cuda()
    matrix[row_indices, col_indices] = 1
    return matrix


def train_tmpgen(train_graphs, gt_graphs, val_graph, val_gt_graph, optimizer, rnn_unit, args, task_id=0, input_E=None, input_hidden=None):
    E = input_E
    hidden = input_hidden
    for epoch in range(args.epoches):
        for train_graph, gt_graph in zip(train_graphs, gt_graphs):
            initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
            
            rnn_unit.train()
            optimizer.zero_grad()

            E, hidden = rnn_unit(train_graph, initial_noise, E, hidden)

            E_matrix = E.view(train_graph.shape[1], -1)
            E_matrix = fill_diagonal_with_ones(E_matrix.clone())

            threshold = 0.1
            condi_threshold = E_matrix < threshold
            # E_matrix[condi_threshold] = 0.0
            
            E = E.detach()
            hidden = tuple([i.detach() for i in hidden])

            mse_loss = F.mse_loss(E_matrix, gt_graph.to(device))
            l1_loss = F.l1_loss(E_matrix, gt_graph.to(device))
            E_matrix_sigmoid = torch.sigmoid(E_matrix).to(device)
            gt_graph_sigmoid = torch.sigmoid(gt_graph).to(device)
            bce_loss = F.binary_cross_entropy(E_matrix_sigmoid, gt_graph_sigmoid)

            loss = mse_loss + l1_loss + args.bce_w * bce_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

    if args.dataset == 'Elec2':
        if task_id + 1 >= 38:
            with torch.no_grad():
                E, hidden = rnn_unit(val_graph, initial_noise, E, hidden)
                E_matrix = E.view(val_graph.shape[1], -1)
                E_matrix.fill_diagonal_(1)     # diagonal elements must be 1

                threshold = 0.1
                condi_threshold = E_matrix < threshold
                E_matrix[condi_threshold] = 0.0

                E = E.detach()
                hidden = tuple([i.detach() for i in hidden])
                mse_loss = F.mse_loss(E_matrix, val_gt_graph.to(device))
                l1_loss = F.l1_loss(E_matrix, val_gt_graph.to(device))
                E_matrix_sigmoid = torch.sigmoid(E_matrix).to(device)
                val_gt_graph_sigmoid = torch.sigmoid(val_gt_graph).to(device)
                bce_loss = F.binary_cross_entropy(E_matrix_sigmoid, val_gt_graph_sigmoid)

                val_loss = mse_loss + l1_loss + bce_loss
                print(f"val_loss = {val_loss:.5f}, mse_loss = {mse_loss:.5f}, l1_loss = {l1_loss:.5f}, bce_loss = {bce_loss:.5f}")
        
    elif args.dataset == 'Moons':
        if task_id + 1 >= 7:
            with torch.no_grad():
                E, hidden = rnn_unit(val_graph, initial_noise, E, hidden)
                E_matrix = E.view(val_graph.shape[1], -1)
                E_matrix.fill_diagonal_(1)     # diagonal elements must be 1

                threshold = 0.0
                condi_threshold = E_matrix < threshold
                E_matrix[condi_threshold] = 0.0

                E = E.detach()
                hidden = tuple([i.detach() for i in hidden])
                mse_loss = F.mse_loss(E_matrix, val_gt_graph.to(device))
                l1_loss = F.l1_loss(E_matrix, val_gt_graph.to(device))
                E_matrix_sigmoid = torch.sigmoid(E_matrix).to(device)
                val_gt_graph_sigmoid = torch.sigmoid(val_gt_graph).to(device)
                bce_loss = F.binary_cross_entropy(E_matrix_sigmoid, val_gt_graph_sigmoid)

                val_loss = mse_loss + l1_loss + bce_loss
                print(f"val_loss = {val_loss:.5f}, mse_loss = {mse_loss:.5f}, l1_loss = {l1_loss:.5f}, bce_loss = {bce_loss:.5f}")
    
    elif args.dataset == 'ONP':
        if task_id + 1 >= 3:
            with torch.no_grad():
                E, hidden = rnn_unit(val_graph, initial_noise, E, hidden)
                E_matrix = E.view(val_graph.shape[1], -1)
                E_matrix.fill_diagonal_(1)     # diagonal elements must be 1

                threshold = 0.1
                condi_threshold = E_matrix < threshold
                E_matrix[condi_threshold] = 0.0

                E = E.detach()
                hidden = tuple([i.detach() for i in hidden])
                mse_loss = F.mse_loss(E_matrix, val_gt_graph.to(device))
                l1_loss = F.l1_loss(E_matrix, val_gt_graph.to(device))
                E_matrix_sigmoid = torch.sigmoid(E_matrix).to(device)
                val_gt_graph_sigmoid = torch.sigmoid(val_gt_graph).to(device)
                bce_loss = F.binary_cross_entropy(E_matrix_sigmoid, val_gt_graph_sigmoid)

                val_loss = mse_loss + l1_loss + bce_loss
                print(f"val_loss = {val_loss:.5f}, mse_loss = {mse_loss:.5f}, l1_loss = {l1_loss:.5f}, bce_loss = {bce_loss:.5f}")

    elif args.dataset == 'Shuttle':
        if task_id + 1 >= 5:
            with torch.no_grad():
                E, hidden = rnn_unit(val_graph, initial_noise, E, hidden)
                E_matrix = E.view(val_graph.shape[1], -1)
                E_matrix.fill_diagonal_(1)     # diagonal elements must be 1

                threshold = 0.01
                condi_threshold = E_matrix < threshold
                E_matrix[condi_threshold] = 0.0

                E = E.detach()
                hidden = tuple([i.detach() for i in hidden])
                mse_loss = F.mse_loss(E_matrix, val_gt_graph.to(device))
                l1_loss = F.l1_loss(E_matrix, val_gt_graph.to(device))
                E_matrix_sigmoid = torch.sigmoid(E_matrix).to(device)
                val_gt_graph_sigmoid = torch.sigmoid(val_gt_graph).to(device)
                bce_loss = F.binary_cross_entropy(E_matrix_sigmoid, val_gt_graph_sigmoid)

                val_loss = mse_loss + l1_loss + bce_loss
                print(f"val_loss = {val_loss:.5f}, mse_loss = {mse_loss:.5f}, l1_loss = {l1_loss:.5f}, bce_loss = {bce_loss:.5f}")

    return E, hidden, rnn_unit


def main(arsgs):
    output_directory='outputs-{}'.format(args.dataset)
    model_directory='models-{}'.format(args.dataset)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)
        
    log('use {} data'.format(args.dataset))
    log('-'*40)
    
    if args.dataset == 'Moons':
        num_tasks=10
        data_size=2
        num_instances=220
    elif args.dataset == 'MNIST':
        num_tasks=11
        data_size=2
        num_instances=200
    elif args.dataset == 'ONP':
        num_tasks=6
        data_size=58
        num_instances=None
    elif args.dataset == 'Elec2':
        num_tasks=41
        data_size=8
        num_instances=None
    elif args.dataset == 'Shuttle':
        num_tasks=8
        data_size=9
        num_instances=None

    # Defining dataloaders for each domain
    dataloaders = dataset_preparation(args, num_tasks, num_instances)
    rnn_unit = RNN(data_size+1, device, args).to(device)   # data_size+1 b/c num of features + y
    
    # Loss and optimizer
    optimizer = torch.optim.Adam(rnn_unit.parameters(), lr=args.learning_rate, weight_decay=0.01)

    starting_time = time.time()
    
    # Training
    if args.dataset == 'Elec2':
        Es, hiddens = [None], [None]
        each_span = 6   # num of data in each span
        for task_id in range(len(true_list_val_graph)):
            start_i = task_id * each_span
            end_i = start_i + each_span
            train_graphs = true_list_train_graph[start_i:end_i]
            gt_graphs = true_list_gt_graph[start_i:end_i]
            val_graph = true_list_val_graph[task_id]
            val_gt_graph = true_list_val_gt_graph[task_id]

            E, hidden, rnn_unit = train_tmpgen(train_graphs, gt_graphs, val_graph, val_gt_graph, optimizer, rnn_unit, args, task_id, Es[-1], hiddens[-1])
            Es.append(E)
            hiddens.append(hidden)
    
    elif args.dataset == 'Moons':
        Es, hiddens = [None], [None]
        each_span = 2   # num of data in each span
        for task_id in range(len(true_list_val_graph)):
            start_i = task_id * each_span
            end_i = start_i + each_span
            train_graphs = true_list_train_graph[start_i:end_i]
            gt_graphs = true_list_gt_graph[start_i:end_i]
            val_graph = true_list_val_graph[task_id]
            val_gt_graph = true_list_val_gt_graph[task_id]

            E, hidden, rnn_unit = train_tmpgen(train_graphs, gt_graphs, val_graph, val_gt_graph, optimizer, rnn_unit, args, task_id, Es[-1], hiddens[-1])
            Es.append(E)
            hiddens.append(hidden)

    elif args.dataset == 'ONP':
        Es, hiddens = [None], [None]
        # each_span = 6   # num of data in each span
        span_num_list = [0, 56, 103, 148, 198, 256, 313]
        for task_id in range(len(true_list_val_graph)):
            start_i = span_num_list[task_id]
            end_i = span_num_list[task_id + 1]
            train_graphs = true_list_train_graph[start_i:end_i]
            gt_graphs = true_list_gt_graph[start_i:end_i]
            val_graph = true_list_val_graph[task_id]
            val_gt_graph = true_list_val_gt_graph[task_id]

            E, hidden, rnn_unit = train_tmpgen(train_graphs, gt_graphs, val_graph, val_gt_graph, optimizer, rnn_unit, args, task_id, Es[-1], hiddens[-1])
            Es.append(E)
            hiddens.append(hidden)

    elif args.dataset == 'Shuttle':
        Es, hiddens = [None], [None]
        each_span = 57   # num of data in each span
        for task_id in range(len(true_list_val_graph)):
            start_i = task_id * each_span
            end_i = start_i + each_span
            train_graphs = true_list_train_graph[start_i:end_i]
            gt_graphs = true_list_gt_graph[start_i:end_i]
            val_graph = true_list_val_graph[task_id]
            val_gt_graph = true_list_val_gt_graph[task_id]

            E, hidden, rnn_unit = train_tmpgen(train_graphs, gt_graphs, val_graph, val_gt_graph, optimizer, rnn_unit, args, task_id, Es[-1], hiddens[-1])
            Es.append(E)
            hiddens.append(hidden)

    ending_time = time.time()
        

if __name__ == "__main__":
    print("Start Training...")
    
    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    main(args)









