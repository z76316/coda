# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd

from torch import nn
import torch.optim as optim

import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse
import pickle

# Import model
from model import RNN, MLP_Elec2, MLP_Moons, MLP_ONP, MLP_Shuttle
# Import functions
from utils import dataset_preparation, make_noise

from torch.utils.data import TensorDataset, DataLoader

import lightgbm as lgb

from sklearn.metrics import accuracy_score, log_loss

# Goggle
from goggle.GoggleModel import GoggleModel

# Synthcity
from synthcity.plugins.core.dataloader import GenericDataLoader

# FFTransformer & TabTransformer
from pytorch_tabular import TabularModel
from pytorch_tabular.models import ( 
    FTTransformerConfig, 
)
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig


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


# log('Is GPU available? {}'.format(torch.cuda.is_available()))
# #print('Is GPU available? {}'.format(torch.cuda.is_available()))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DomainGen_Graph")

datasets = ['ONP', 'Moons', 'MNIST', 'Elec2', 'Shuttle']
parser.add_argument("--dataset", default="Elec2", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))

# Hyper-parameters
parser.add_argument("--noise_dim", default=16, type=float,
                    help="the dimension of the LSTM input noise.")
parser.add_argument("--num_rnn_layer", default=1, type=int,
                    help="the number of RNN hierarchical layers.")
parser.add_argument("--latent_dim", default=16, type=int,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--hidden_dim", default=8, type=int,
                    help="the latent dimension of RNN variables.")
parser.add_argument("--noise_type", choices=["Gaussian", "Uniform"], default="Gaussian",
                    help="The noise type to feed into the generator.")

parser.add_argument("--num_workers", default=0, type=int,
                    help="the number of threads for loading data.")
parser.add_argument("--epoches", default=20, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--batch_size", default=64, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="the unified learning rate for each single task.")
parser.add_argument("--dropout_rate", default=0.5, type=float,
                    help="the dropout rate or LSTM.")

parser.add_argument("--is_test", default=True, type=bool,
                    help="if this is a testing period.")

parser.add_argument("--gpu_id", default=0, type=int,
                    help="the gpu id of for the exp.")

parser.add_argument("--encoder_dim", default=64, type=int,
                    help="the encoder dim of GOGGLE.")
parser.add_argument("--encoder_l", default=3, type=int,
                    help="the encoder layers of GOGGLE.")
parser.add_argument("--decoder_dim", default=72, type=int,
                    help="the decoder dim of GOGGLE.")
parser.add_argument("--decoder_l", default=3, type=int,
                    help="the decoder layers of GOGGLE.")
parser.add_argument("--threshold", default=0.1, type=float,
                    help="the threshold of GOGGLE.")
parser.add_argument("--goggle_beta", default=1e-1, type=float,
                    help="the beta of GOGGLE.")
parser.add_argument("--goggle_lr", default=2e-2, type=float,
                    help="the learning rate of GOGGLE.")
parser.add_argument("--goggle_batch_size", default=128, type=int,
                    help="the batch size of GOGGLE.")
parser.add_argument("--patience", default=50, type=int,
                    help="the patience of GOGGLE.")
parser.add_argument("--goggle_epoches", default=1000, type=int,
                    help="the epoches of GOGGLE.")
parser.add_argument("--goggle_logging", default=100, type=int,
                    help="the step for each log of GOGGLE.")

tune_data_models = ['MLP', 'LightGBM', 'FTTransformer']
parser.add_argument("--tune_data_model", default="MLP", type=str,
                    help="one of: {}".format(", ".join(sorted(tune_data_models))))

pred_models = ['MLP', 'LightGBM', 'FTTransformer']
parser.add_argument("--pred_model", default="FTTransformer", type=str,
                    help="one of: {}".format(", ".join(sorted(pred_models))))

# MLP
parser.add_argument("--MLP_lr", default=1e-4, type=float,
                    help="the lr of MLP.")

# LightGBM
parser.add_argument("--LG_lr", default=0.05, type=float,
                    help="the lr of LightGBM.")
parser.add_argument("--LG_ff", default=1.0, type=float,
                    help="the feature_fraction of LightGBM.")
parser.add_argument("--LG_bfrac", default=1.0, type=float,
                    help="the bagging_fraction of LightGBM.")
parser.add_argument("--LG_bfreq", default=5, type=int,
                    help="the bagging_freq of LightGBM.")
parser.add_argument("--LG_l1", default=0.0, type=float,
                    help="the lambda_l1 of LightGBM.")
parser.add_argument("--LG_l2", default=0.0, type=float,
                    help="the lambda_l2 of LightGBM.")
parser.add_argument("--is_linear_tree", default=False, type=bool,
                    help="if the model at leaf is a linear.")
parser.add_argument("--LG_lambda", default=0.0, type=float,
                    help="the linear_lambda of LightGBM.")
parser.add_argument("--LG_leaves", default=31, type=int,
                    help="the max num of LightGBM leaves.")

# FFTransformer & TabTransformer
parser.add_argument("--T_batch", default=64, type=int,
                    help="the batch size of Transformer.")
parser.add_argument("--T_lr", default=8e-5, type=float,
                    help="the lr of Transformer.")
parser.add_argument("--T_dropout", default=0.1, type=float,
                    help="the lr of Transformer.")
parser.add_argument("--T_i_dim", default=128, type=int,
                    help="the input dim of Transformer.")
parser.add_argument("--T_n_head", default=8, type=int,
                    help="the number of att head in Transformer.")
parser.add_argument("--T_att_blocks", default=8, type=int,
                    help="the number of att blocks in Transformer.")
parser.add_argument("--T_attn_dropout", default=0.5, type=float,
                    help="the attn_dropout of Transformer.")

args = parser.parse_args()

# log('Is GPU available? {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
    device = f"cuda:{args.gpu_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
else:
    device = "cpu"


save_dir = f"data/{args.dataset}"
path_true_list_train_graph = f"{args.dataset}_true_list_train_graph_lr-0.01.pkl"
path_true_list_gt_graph = f"{args.dataset}_true_list_gt_graph_lr-0.01.pkl"
path_true_list_val_graph = f"{args.dataset}_true_list_val_graph_lr-0.01.pkl"
path_true_list_val_gt_graph = f"{args.dataset}_true_list_val_gt_graph_lr-0.01.pkl"
path_pre_tmp_val_graph = f"{args.dataset}_pre_tmp_val_graph.pkl"
path_tmp_val_graph = f"{args.dataset}_tmp_val_graph.pkl"

if args.tune_data_model != "MLP":
    path_gen_val_data = f"{args.dataset}_gen_val_data_{args.tune_data_model}_ed-{args.encoder_dim}_el-{args.encoder_l}_dd-{args.decoder_dim}_dl-{args.decoder_l}_b-{args.goggle_beta}_lr_{args.goggle_lr}.pkl"
    path_gen_test_data = f"{args.dataset}_gen_test_data_{args.tune_data_model}_ed-{args.encoder_dim}_el-{args.encoder_l}_dd-{args.decoder_dim}_dl-{args.decoder_l}_b-{args.goggle_beta}_lr_{args.goggle_lr}.pkl"
else:
    path_gen_val_data = f"{args.dataset}_gen_val_data_ed-{args.encoder_dim}_el-{args.encoder_l}_dd-{args.decoder_dim}_dl-{args.decoder_l}_b-{args.goggle_beta}_lr_{args.goggle_lr}.pkl"
    path_gen_test_data = f"{args.dataset}_gen_test_data_ed-{args.encoder_dim}_el-{args.encoder_l}_dd-{args.decoder_dim}_dl-{args.decoder_l}_b-{args.goggle_beta}_lr_{args.goggle_lr}.pkl"

with open(os.path.join(save_dir, path_true_list_train_graph), 'rb') as f:
    true_list_train_graph = pickle.load(f)

with open(os.path.join(save_dir, path_true_list_gt_graph), 'rb') as f:
    true_list_gt_graph = pickle.load(f)

with open(os.path.join(save_dir, path_true_list_val_graph), 'rb') as f:
    true_list_val_graph = pickle.load(f)

with open(os.path.join(save_dir, path_true_list_val_gt_graph), 'rb') as f:
    true_list_val_gt_graph = pickle.load(f)

with open(os.path.join(save_dir, path_pre_tmp_val_graph), 'rb') as f:
    pre_tmp_val_graph = pickle.load(f)

with open(os.path.join(save_dir, path_tmp_val_graph), 'rb') as f:
    tmp_val_graph = pickle.load(f)


def gen_test_data(cur_dataloader, test_dataloader, tmp_val_graph, args):

    # ========== Temporal Generator ==========
    # Initialize lists to store batches
    data_batches = []
    label_batches = []
    
    # Iterate through the DataLoader and store batches
    for batch_data, batch_labels in cur_dataloader:
        data_batches.append(batch_data)
        label_batches.append(batch_labels)

    # Concatenate the batches into single tensors
    X = torch.cat(data_batches, dim=0)
    Y = torch.cat(label_batches, dim=0)
    Y = Y.unsqueeze(1)
            
    X = torch.cat((X, Y), dim=1)

    # Convert the NumPy array to a pandas DataFrame
    df_X = pd.DataFrame(X)

    # ======== downsampling to make label 1:1 ========
    if args.dataset == "Shuttle":
        min_count = df_X.iloc[:, -1].value_counts().min()
        downsampled_df = df_X.groupby(df_X.iloc[:, -1]).apply(lambda x: x.sample(min_count)).reset_index(drop=True)
        df_X = downsampled_df

    columns = df_X.columns.tolist()
    columns[-1] = "target"
    df_X.columns = columns

    # Convert integer column names to strings
    df_X.columns = df_X.columns.astype(str)

    # graph_prior & prior_mask
    graph_prior = torch.Tensor(tmp_val_graph).detach()
    prior_mask = torch.Tensor(tmp_val_graph).detach()
    # graph_prior = None
    # prior_mask = None

    # GOGGLE
    gen = GoggleModel(
        ds_name=args.dataset,
        input_dim=df_X.shape[1],
        encoder_dim=args.encoder_dim,   #64
        encoder_l=args.encoder_l,   #2
        het_encoding=True,
        decoder_dim=args.decoder_dim,   #64
        decoder_l=args.decoder_l,   #2
        threshold=args.threshold,   #0.1
        decoder_arch="gcn",   #gcn
        graph_prior=graph_prior,     #None
        prior_mask=prior_mask,     #None
        device=device,
        beta=args.goggle_beta,   #0.1
        learning_rate=args.goggle_lr,   #0.01
        seed=0,
        batch_size=args.goggle_batch_size,
        epochs=args.goggle_epoches,
        logging=args.goggle_logging,
        patience=args.patience
    )
    
    gen.fit(df_X)
    
    # Initialize lists to store batches
    test_data_batches = []
    test_label_batches = []
    
    # Iterate through the DataLoader and store batches
    for batch_data, batch_labels in test_dataloader:
        test_data_batches.append(batch_data)
        test_label_batches.append(batch_labels)

    # Concatenate the batches into single tensors
    X = torch.cat(test_data_batches, dim=0)
    Y = torch.cat(test_label_batches, dim=0)
    Y = Y.unsqueeze(1)
            
    X = torch.cat((X, Y), dim=1)

    # Convert the NumPy array to a pandas DataFrame
    df_X_test = pd.DataFrame(X)

    columns = df_X_test.columns.tolist()
    columns[-1] = "target"
    df_X_test.columns = columns

    # Convert integer column names to strings
    df_X_test.columns = df_X_test.columns.astype(str)

    df_X_synth = gen.sample(df_X_test)
    df_y_synth = df_X_synth.pop('target')

    X_train_synth = torch.tensor(df_X_synth.values, dtype=torch.float32, requires_grad=True).to(device)
    y_train_synth = torch.tensor(df_y_synth.values, dtype=torch.float32, requires_grad=True).to(device)


    df_y_test = df_X_test.pop('target')


    if args.tune_data_model == "LightGBM":
        X_test = df_X_test.to_numpy()
        y_test = df_y_test.to_numpy()

        X_train_synth_cpu = X_train_synth.detach().cpu().numpy()
        y_train_synth_cpu = y_train_synth.detach().cpu().numpy()
        
        lgb_train = lgb.Dataset(X_train_synth_cpu, y_train_synth_cpu,
                                free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                               free_raw_data=False)

        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_error', 'binary_logloss'],
            'num_leaves': 31,
            'learning_rate': args.LG_lr,
            'feature_fraction': args.LG_ff,
            'bagging_fraction': args.LG_bfrac,
            'bagging_freq': args.LG_bfreq,
            'lambda_l1': args.LG_l1,
            'lambda_l2': args.LG_l2,
            'linear_tree': args.is_linear_tree,
            'linear_lambda': args.LG_lambda,
            'verbose': -1
        }

        # generate feature names
        feature_name = [f'feature_{col}' for col in range(X_train_synth.shape[1])]

        evals_result = {}
        num_epochs = 2000
        model = lgb.train(params,
                lgb_train,
                num_boost_round=num_epochs,
                valid_sets=[lgb_train, lgb_eval],  # eval training data
                feature_name=feature_name,
                callbacks=[
                    # lgb.log_evaluation(100),
                    lgb.record_evaluation(evals_result)
                ])

        y_hat_proba = model.predict(X_test)

        y_hat = (y_hat_proba > 0.5).astype(int)
        acc = accuracy_score(y_test, y_hat)
        loss = log_loss(y_test, y_hat_proba)

        val_loss_list = evals_result['training']['binary_logloss']
        test_loss_list = evals_result['valid_1']['binary_logloss']
        val_acc_list = [1 - b_error for b_error in evals_result['training']['binary_error']]
        test_acc_list = [1 - b_error for b_error in evals_result['valid_1']['binary_error']]
        
        best_acc_test = max(test_acc_list)
        best_index = test_acc_list.index(best_acc_test)
        best_loss = test_loss_list[best_index]

        print(f"Epoch [{best_index + 1}/{num_epochs}], Loss: {best_loss:.6f}, acc_test = {best_acc_test:.6f}")
        print()

    elif args.tune_data_model == "FTTransformer":
        print(f"========== dataset: {args.dataset}, T_lr: {args.T_lr}, T_i_dim: {args.T_i_dim}, T_n_head: {args.T_n_head}, T_att_blocks: {args.T_att_blocks}, T_attn_dropout: {args.T_attn_dropout}, T_batch: {args.T_batch} ==========")

        data_config = DataConfig(
            target=['target'], #target should always be a list.
            continuous_cols=columns[:-1],
            # categorical_cols=cat_cols,
        )

        trainer_config = TrainerConfig(
            # auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
            batch_size=args.T_batch,
            max_epochs=500,
            early_stopping="valid_loss", # Monitor valid_loss for early stopping
            early_stopping_mode = "min", # Set the mode as min because for val_loss, lower is better
            early_stopping_patience=50, # No. of epochs of degradation training will wait before terminating
            # checkpoints="valid_loss", # Save best checkpoint monitoring val_loss
            checkpoints=None, # Save best checkpoint monitoring val_loss
            load_best=True, # After training, load the best checkpoint
            # checkpoints_path="saved_models"
        )

        optimizer_config = OptimizerConfig()

        head_config = LinearHeadConfig(
            layers="", # No additional layer in head, just a mapping layer to output_dim
            dropout=args.T_dropout,
            initialization="kaiming"
        ).__dict__ # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)
        
        model_config = FTTransformerConfig(
            task="classification",
            learning_rate = args.T_lr,   #1e-3
            head = "LinearHead", #Linear Head
            head_config = head_config, # Linear Head Config
            input_embed_dim = args.T_i_dim,   #32
            num_heads = args.T_n_head,   #8
            num_attn_blocks = args.T_att_blocks,   #6
            attn_dropout=args.T_attn_dropout,   #0.1
        )

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        y_train_synth = y_train_synth.unsqueeze(1)
        
        df_X_train_synth = pd.DataFrame(X_train_synth)
        df_y_train_synth = pd.DataFrame(y_train_synth)
        df_train_synth = pd.concat([df_X_train_synth, df_y_train_synth], axis=1)
        df_test = pd.concat([df_X_test, df_y_test], axis=1)

        df_train_synth.columns = columns
        df_test.columns = columns

        tabular_model.fit(train=df_train_synth)
        tabular_model.evaluate(df_test, verbose=True)


    elif args.pred_model == "MLP":
        X_test = torch.tensor(df_X_test.values, dtype=torch.float32).to(device)
        y_test = torch.tensor(df_y_test.values, dtype=torch.float32).to(device)

        mlp = ""
        if args.dataset == "Elec2":
            mlp = MLP_Elec2(X_train_synth.shape[1]).to(device)
        elif args.dataset == "Moons":
            mlp = MLP_Moons(X_train_synth.shape[1]).to(device)
        elif args.dataset == "ONP":
            mlp = MLP_ONP(X_train_synth.shape[1]).to(device)
        elif args.dataset == "Shuttle":
            mlp = MLP_Shuttle(X_train_synth.shape[1]).to(device)
        optimizer = optim.Adam(mlp.parameters(), lr=args.MLP_lr, weight_decay=0.01)  # Adam optimizer lr=0.0005

        if args.dataset == "Shuttle":
            batch_size = args.batch_size  # Define your batch size
            train_dataset = TensorDataset(X_train_synth, y_train_synth)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_epoch, best_loss, best_acc_test = 0, 1, 0
        num_epochs = 1000
        for epoch in range(num_epochs):
            if args.dataset == "Shuttle":
                for batch_X, batch_Y in train_loader:
                    y_hat = mlp(batch_X)
                    loss = F.binary_cross_entropy(y_hat.squeeze(-1), batch_Y)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                # Forward pass
                y_hat = mlp(X_train_synth)
                loss = F.binary_cross_entropy(y_hat.squeeze(-1), y_train_synth)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % 1 == 0:
                with torch.no_grad():
                    if args.dataset == 'Shuttle':
                        y_hat_test = mlp(X_test)
                        y_hat_test = torch.as_tensor((y_hat_test.detach() - 0.5) > 0).float()
                        acc_test = (y_hat_test.squeeze(-1) == y_test).float().sum() / y_hat_test.shape[0]
                    else:
                        y_hat_test = mlp(X_test)
                        y_hat_test = torch.as_tensor((y_hat_test.detach() - 0.5) > 0).float()
                        acc_test = (y_hat_test.squeeze(-1) == y_test).float().sum() / y_hat_test.shape[0]

                    if acc_test > best_acc_test:
                        best_epoch, best_loss, best_acc_test = epoch + 1, loss.item(), acc_test
                        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, acc_test = {acc_test:.6f}')

        
        print(f"Epoch [{best_epoch}/{num_epochs}], Loss: {best_loss:.6f}, acc_test = {best_acc_test:.6f}")
        print()

    # save the CODA-generated data
    if args.tune_data_model != "MLP":
        # path_gen_val_data = f"{args.dataset}_gen_val_data_{args.tune_data_model}_ed-{args.encoder_dim}_el-{args.encoder_l}_dd-{args.decoder_dim}_dl-{args.decoder_l}_b-{args.goggle_beta}_lr_{args.goggle_lr}.pkl"
        path_gen_test_data = f"{args.dataset}_gen_test_data_{args.tune_data_model}_ed-{args.encoder_dim}_el-{args.encoder_l}_dd-{args.decoder_dim}_dl-{args.decoder_l}_b-{args.goggle_beta}_lr_{args.goggle_lr}.pkl"
    else:
        # path_gen_val_data = f"{args.dataset}_gen_val_data_ed-{args.encoder_dim}_el-{args.encoder_l}_dd-{args.decoder_dim}_dl-{args.decoder_l}_b-{args.goggle_beta}_lr_{args.goggle_lr}.pkl"
        path_gen_test_data = f"{args.dataset}_gen_test_data_ed-{args.encoder_dim}_el-{args.encoder_l}_dd-{args.decoder_dim}_dl-{args.decoder_l}_b-{args.goggle_beta}_lr_{args.goggle_lr}.pkl"

    # with open(os.path.join(save_dir, path_gen_test_data), 'wb') as f:
    #     pickle.dump((X_train_synth, y_train_synth), f)


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

    print(f"========== Generate CODA data ==========")
    print(f"=== dataset: {args.dataset}, en_dim: {args.encoder_dim}, en_l: {args.encoder_l}, de_dim: {args.decoder_dim}, de_l: {args.decoder_l}, thre: {args.threshold}, goggle_batch_size:{args.goggle_batch_size}, beta: {args.goggle_beta}, goggle_lr: {args.goggle_lr} ===")
    if args.pred_model == "MLP":
        print(f"=== tune_data_model: {args.tune_data_model}, MLP_lr: {args.MLP_lr} ===")
    elif args.pred_model == "LightGBM":
        print(f"=== tune_data_model: {args.tune_data_model}, LG_lr: {args.LG_lr}, LG_ff: {args.LG_ff}, LG_bfrac: {args.LG_bfrac}, LG_bfreq: {args.LG_bfreq}, LG_l1: {args.LG_l1}, LG_l2: {args.LG_l2}, LG_lambda: {args.LG_lambda}, LG_leaves: {args.LG_leaves} ===")
    elif args.pred_model == "FTTransformer":
        print(f"=== tune_data_model: {args.tune_data_model}, T_lr: {args.T_lr}, T_i_dim: {args.T_i_dim}, T_n_head: {args.T_n_head}, T_att_blocks: {args.T_att_blocks}, T_attn_dropout: {args.T_attn_dropout}, T_batch: {args.T_batch} ===")
    
    starting_time = time.time()

    gen_test_data(dataloaders[-2], dataloaders[-1], tmp_val_graph, args)

    ending_time = time.time()

    print("Training time:", ending_time - starting_time)
        

if __name__ == "__main__":
    print("Start Training...")
    
    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    main(args)









