'''
* DNDC
* 2020-1-10
'''

import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsn
import utils

from scipy import sparse as sp
import scipy.io as sio
import csv

from TNDconf import TNDconf

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='Flickr')  # BlogCatalog # Flickr peerRead
parser.add_argument('--path', type=str, default='/u/jm3mr/simulate/Flickr_shuffle/')
#parser.add_argument('--path', type=str, default='/u/jm3mr/simulate/peerRead/')

parser.add_argument('--alpha', type=float, default= 1, help='reverse layer, multiply by a constant') # alpha should > 0
parser.add_argument('--beta', type=float, default= 1, help='weight of treatment prediction loss.')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Initial learning rate.')
parser.add_argument('--h_dim', type=int, default=50,
                    help='dim of hidden units.')
parser.add_argument('--z_dim', type=int, default=50,
                    help='dim of hidden confounders.')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--normy', type=int, default=1)
parser.add_argument('--n_layers_gcn', type=int, default=1)
parser.add_argument('--n_out', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--P', type=int, default=3)

parser.add_argument('--wass', type=float, default=1e-4)

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def load_data(path, name='BlogCatalog', exp_id='0'):
    data = sio.loadmat(path + name + exp_id + '.mat')

    C_list = data['T']
    Y1_true_list = data['Y1']
    Y0_true_list = data['Y0']
    idx_trn = data['trn_idx'][0]
    idx_val = data['val_idx'][0]
    idx_tst = data['tst_idx'][0]

    # load
    X = data['X'][0]

    Z_init = torch.zeros(X[0].shape[0], args.h_dim)

    X_list = []
    for t in range(len(X)):
        n_x = len(X)
        xt = X[t]
        X_list.append(torch.tensor(X[t].todense(), dtype=torch.float32))

    # A
    sparse_A_list = []
    dense_A_list = []

    A = data['A'][0]
    for t in range(len(A)):
        dense_A_list.append(torch.tensor(A[t].todense()))
        A[t] = sp.csr_matrix(A[t])
        A[t] = utils.sparse_mx_to_torch_sparse_tensor(A[t])
        sparse_A_list.append(A[t])

    C_list = [torch.FloatTensor(C) for C in C_list]
    Y1_true_list = [torch.FloatTensor(y1) for y1 in Y1_true_list]
    Y0_true_list = [torch.FloatTensor(y0) for y0 in Y0_true_list]
    idx_trn = torch.LongTensor(idx_trn)
    idx_val = torch.LongTensor(idx_val)
    idx_tst = torch.LongTensor(idx_tst)

    idx_trn_list = []
    idx_val_list = []
    idx_tst_list = []
    for t in range(len(A)):
        idx_trn_list.append(idx_trn)
        idx_val_list.append(idx_val)
        idx_tst_list.append(idx_tst)

    Z_init = torch.FloatTensor(Z_init)

    return X_list, sparse_A_list, dense_A_list, C_list, Y1_true_list, Y0_true_list, idx_trn_list, idx_val_list, idx_tst_list, Z_init


def compute_loss(y_pred_list, y1_true_list, y0_true_list, rep_list, C_list, ps_pred_list, idx_trn_list, idx_val_list,
                 beta=1.0):
    beta = torch.FloatTensor([beta]).to(device)  # ps

    loss_train = 0.0
    T = len(y_pred_list)
    loss_mse = torch.nn.MSELoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    for t in range(T):
        rep = rep_list[t]
        C = C_list[t]
        Y1 = y1_true_list[t]
        Y0 = y0_true_list[t]
        yf_pred = y_pred_list[t]
        ps_pred = ps_pred_list[t]

        idx_train = idx_trn_list[t]
        idx_val = idx_val_list[t]

        # potential outcome prediction
        YF = torch.where(C > 0, Y1, Y0)
        YCF = torch.where(C > 0, Y0, Y1)

        # norm y
        if args.normy:
            ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
            YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys
        else:
            YFtr = YF[idx_train]
            YFva = YF[idx_val]

        loss_y = loss_mse(yf_pred[idx_train], YFtr)
        loss_c = loss_ce(ps_pred[idx_train], C[idx_train].long())

        loss_train_t = loss_y + beta * loss_c
        loss_train += loss_train_t

    return loss_train


def evaluate(y1_pred, y0_pred, y1_true, y0_true, C, ps_pred, idx_train, idx_test):
    loss_mse = torch.nn.MSELoss()

    # potential outcome prediction
    YF = torch.where(C > 0, y1_true, y0_true)
    YCF = torch.where(C > 0, y0_true, y1_true)

    # norm y
    if args.normy:
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    ITE_pred = y1_pred - y0_pred

    pehe = torch.sqrt(loss_mse(ITE_pred[idx_test], (y1_true - y0_true)[idx_test]))
    mae_ate = torch.abs(
        torch.mean(ITE_pred[idx_test]) - torch.mean((y1_true - y0_true)[idx_test]))

    # acc of treatment prediction
    C_pred = ps_pred.argmax(dim=1)
    corrects = (C_pred[idx_test] == C[idx_test])
    acc_t = corrects.sum().float() / float(len(idx_test))

    return pehe, mae_ate, acc_t


# training the model
def train(epoch, X_list, adj_time_list, C_list, Y1_true_list, Y0_true_list, idx_trn_list, idx_val_list, idx_tst_list,
          Z_init, model, optimizer):
    time_begin = time.time()

    model.train()
    T = len(X_list)

    seq_start = 0
    seq_end = T

    print("start training!")

    C_list_CF = [1 - c for c in C_list]

    for k in range(epoch):  # epoch
        optimizer.zero_grad()

        # forward
        y1_pred_list, y0_pred_list, rep_list, ps_pred_list, hidden_rep = \
            model(X_list[seq_start:seq_end], adj_time_list[seq_start:seq_end], C_list[seq_start:seq_end],
                  hidden_in=Z_init)  # node attributes, graph, treatments

        yf_pred_list = [torch.where(C_list[t] > 0, y1_pred_list[t], y0_pred_list[t]) for t in range(T)]
        ycf_pred_list = [torch.where(C_list[t] > 0, y0_pred_list[t], y1_pred_list[t]) for t in range(T)]

        # loss
        loss_train = compute_loss(yf_pred_list, Y1_true_list, Y0_true_list, rep_list, C_list, ps_pred_list,
                                  idx_trn_list, idx_val_list, args.beta)

        loss_train.backward()
        optimizer.step()

        nn.utils.clip_grad_norm(model.parameters(), args.clip)

        # compute the pehe and mae of all the time stamps
        pehe_val_list = []
        mae_ate_val_list = []
        acc_t_val_list = []
        pehe_tst_list = []
        mae_ate_tst_list = []
        acc_t_tst_list = []
        acc_t_trn_list = []

        if k % 10 == 0:
            for select_t in range(T):
                y1_true = Y1_true_list[select_t]
                y0_true = Y0_true_list[select_t]
                yf_pred = yf_pred_list[select_t]
                ycf_pred = ycf_pred_list[select_t]
                C = C_list[select_t]
                ps_pred = ps_pred_list[select_t]

                idx_train = idx_trn_list[select_t]
                idx_val = idx_val_list[select_t]
                idx_tst = idx_tst_list[select_t]

                y1_pred, y0_pred = torch.where(C > 0, yf_pred, ycf_pred), torch.where(C > 0, ycf_pred, yf_pred)

                #
                pehe_trn, mae_ate_trn, acc_t_trn = evaluate(y1_pred, y0_pred, y1_true, y0_true, C, ps_pred, idx_train,
                                                                             idx_train)
                # val
                pehe_val, mae_ate_val, acc_t_val = evaluate(y1_pred, y0_pred, y1_true, y0_true, C, ps_pred, idx_train,
                                                            idx_val)
                # test
                pehe_tst, mae_ate_tst, acc_t_tst = evaluate(y1_pred, y0_pred, y1_true, y0_true, C, ps_pred, idx_train,
                                                            idx_tst)

                acc_t_trn_list.append(acc_t_trn)
                pehe_val_list.append(pehe_val.item())
                mae_ate_val_list.append(mae_ate_val.item())
                acc_t_val_list.append(acc_t_val)
                pehe_tst_list.append(pehe_tst.item())
                mae_ate_tst_list.append(mae_ate_tst.item())
                acc_t_tst_list.append(acc_t_tst)

            print('Epoch: {:04d}'.format(k + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(sum(acc_t_trn_list) / len(acc_t_trn_list)),
                  'ave_pehe_val: {:.4f}'.format(sum(pehe_val_list) / len(pehe_val_list)),
                  'ave_mae_ate_val: {:.4f}'.format(sum(mae_ate_val_list) / len(mae_ate_val_list)),
                  'acc_t_val: {:.4f}'.format(sum(acc_t_val_list) / len(acc_t_val_list)),
                  'ave_pehe_tst: {:.4f}'.format(sum(pehe_tst_list) / len(pehe_tst_list)),
                  'ave_mae_ate_tst: {:.4f}'.format(sum(mae_ate_tst_list) / len(mae_ate_tst_list)),
                  'acc_t_tst: {:.4f}'.format(sum(acc_t_tst_list) / len(acc_t_tst_list)),
                  'time: {:.4f}s'.format(time.time() - time_begin))

            test(X_list, adj_time_list, C_list, Y1_true_list, Y0_true_list, idx_trn_list, idx_tst_list, model, Z_init)
            model.train()


def test(X_list, adj_time_list, C_list, Y1_true_list, Y0_true_list, idx_trn_list, idx_tst_list, model, Z_init,
         saving=False):
    model.eval()

    T = len(X_list)
    seq_start = 0
    seq_end = T

    C_list_CF = [1 - c for c in C_list]

    y1_pred_list, y0_pred_list, rep_list, ps_pred_list, hidden_rep = model(X_list[seq_start:seq_end],
                                                                           adj_time_list[seq_start:seq_end],
                                                                           C_list[seq_start:seq_end], hidden_in=Z_init)

    yf_pred_list = [torch.where(C_list[t] > 0, y1_pred_list[t], y0_pred_list[t]) for t in range(T)]
    ycf_pred_list = [torch.where(C_list[t] > 0, y0_pred_list[t], y1_pred_list[t]) for t in range(T)]

    # compute the pehe and mae of all the time stamps
    pehe_val_list = []
    mae_ate_val_list = []
    acc_t_val_list = []
    pehe_tst_list = []
    mae_ate_tst_list = []
    acc_t_tst_list = []

    for select_t in range(T):
        rep = rep_list[select_t]

        y1_true = Y1_true_list[select_t]
        y0_true = Y0_true_list[select_t]
        yf_pred = yf_pred_list[select_t]
        ycf_pred = ycf_pred_list[select_t]
        C = C_list[select_t]
        ps_pred = ps_pred_list[select_t]

        idx_train = idx_trn_list[select_t]
        idx_val = idx_val_list[select_t]
        idx_tst = idx_tst_list[select_t]

        y1_pred, y0_pred = torch.where(C > 0, yf_pred, ycf_pred), torch.where(C > 0, ycf_pred, yf_pred)

        # val
        pehe_val, mae_ate_val, acc_t_val = evaluate(y1_pred, y0_pred, y1_true, y0_true, C, ps_pred, idx_train, idx_val)
        # test
        pehe_tst, mae_ate_tst, acc_t_tst = evaluate(y1_pred, y0_pred, y1_true, y0_true, C, ps_pred, idx_train, idx_tst)

        pehe_val_list.append(pehe_val.item())
        mae_ate_val_list.append(mae_ate_val.item())
        acc_t_val_list.append(acc_t_val)
        pehe_tst_list.append(pehe_tst.item())
        mae_ate_tst_list.append(mae_ate_tst.item())
        acc_t_tst_list.append(acc_t_tst)

    print('test results: ',
          'ave_pehe_tst: {:.4f}'.format(sum(pehe_tst_list) / len(pehe_tst_list)),
          'ave_mae_ate_tst: {:.4f}'.format(sum(mae_ate_tst_list) / len(mae_ate_tst_list)),
          'ave_acc_t_tst:{:.4f}'.format(sum(acc_t_tst_list) / len(acc_t_tst_list)))


if __name__ == '__main__':
    t_begin = time.time()

    for i_exp in range(0, 10):  # 10 runs of experiments
        # load data
        X_list, adj_time_list, adj_orig_dense_list, C_list, Y1_true_list, Y0_true_list, idx_trn_list, idx_val_list, idx_tst_list, Z_init = load_data(
            args.path, args.dataset, str(i_exp))
        print('finished data loading from: ', args.path, ' dataset: ', args.dataset, ' i_exp: ', i_exp)

        # set model
        x_dim = X_list[0].shape[1]
        model = TNDconf(x_dim, args.h_dim, args.z_dim, args.n_layers_gcn, args.n_out, args.dropout, args.alpha, args.P)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # cuda
        if args.cuda:
            model = model.to(device)
            X_list = [x.to(device) for x in X_list]
            adj_time_list = [e.to(device) for e in adj_time_list]
            C_list = [c.to(device) for c in C_list]
            Y1_true_list = [y1.to(device) for y1 in Y1_true_list]
            Y0_true_list = [y0.to(device) for y0 in Y0_true_list]
            idx_trn_list = [id.to(device) for id in idx_trn_list]
            idx_val_list = [id.to(device) for id in idx_val_list]
            idx_tst_list = [id.to(device) for id in idx_tst_list]

        # training
        train(args.epochs, X_list, adj_time_list, C_list, Y1_true_list, Y0_true_list, idx_trn_list, idx_val_list,
              idx_tst_list, Z_init, model, optimizer)
        test(X_list, adj_time_list, C_list, Y1_true_list, Y0_true_list, idx_trn_list, idx_tst_list, model, Z_init,
             False)

        break  # !!!!!!!!!!!!!

    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
