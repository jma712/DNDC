'''
    The simulation process on dynamic graphs
    created on 2019-12-12
'''

from __future__ import division

import scipy.io as scio
import scipy.sparse as sp
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc, rcParams
from sklearn.manifold import TSNE as tsn

from sklearn.decomposition import LatentDirichletAllocation
from scipy import sparse
from utils_new import *
import os
import pickle

import math

random.seed(111)
np.random.seed(111)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def data_load(dataName, path):
    '''
    load the data
    :return: features and network,
        -features: a list of ndarrays, each elem is a N x dx ndarray; T x N x dx
        -network: a list of ndarrays, each elem is a N x N adjencency matrix
    '''
    if dataName == 'peerRead':
        feature_list, network_list = loadData_peer(path)
    elif dataName == 'Flickr':
        feature_list, network_list = loadData_flickr3(path)
    elif dataName == 'BlogCatalog':
        feature_list, network_list = loadData_flickr3(path)

    return feature_list, network_list


def simulate_peer(data_path):
    root_path = '../dataset/' + 'simulate_new/PeerRead/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    # read the loaded files
    data = scio.loadmat(data_path)
    feature_list = data['X'][0]
    network_list = data['A'][0]
    real_treat_list = data['orin_treat']

    feature_list = [f.toarray() for f in feature_list]
    network_list = [net.toarray() for net in network_list]

    # degree
    n = feature_list[0].shape[0]
    A_dense = network_list[-1]
    if is_symmetric(A_dense):
        print("Symmetric")
    degree_node = np.zeros(n)
    for i in range(n):
        degree_node[i] = len(np.nonzero(A_dense[i])[0])
    ave_degree = np.average(degree_node)
    degree_node.sort()
    degree_node = degree_node[::-1]

    print('degree distri:', degree_node, ' average degree:', ave_degree)


    #
    # reduce by var
    dim_x_reduce = 50
    feature_t_reduced_list = []
    for t in range(len(feature_list)):
        observed_dim = np.argsort(np.var(feature_list[t], axis=0)).reshape(-1)
        observed_dim = observed_dim[-dim_x_reduce:]
        feature_t_reduced = feature_list[t][:, observed_dim]
        feature_t_reduced_list.append(feature_t_reduced)

    # save to file
    scio.savemat(root_path + 'peerRead_feat_dense' + '.mat', {
        'X': feature_list  # do not record the init time stamp
    })
    print("save features done")

    # parameters
    kappa1 = 10
    kappa2 = 1

    B = 0.01
    att = "something"
    kernel = "linear"  # linear or rbf
    saving = False
    saving_all = True
    row_norm_AZ = False
    row_norm_A = False
    norm_z = False
    z_scale = 0.3  # scale of z, to prevent overflow

    name = 'peerRead'

    P = 3  # p-order
    lambda1 = 0.3  # history
    lambda2 = 0.3  # network
    lambda3 = 0.3  # x->z

    trn_rate = 0.6
    val_rate = 0.2
    tst_rate = 0.2

    if kappa2 != 0.1:
        extra_str = str(kappa2)
    else:
        extra_str = ''

    if att != "":
        if kernel == "linear":
            extra_str += 's'
        elif kernel == "rbf":
            extra_str += 'r'
        elif kernel == "poly":
            extra_str += 'n'  # stands for non-linear

    X = feature_list[0]
    A0 = network_list[0]
    A_dense = A0

    n = X.shape[0]
    dx = X.shape[1]

    if row_norm_A:
        A_dense = row_norm(A_dense)

    # Z, AZ = initialization_peer(X, A_dense, row_norm_AZ)
    Z = X
    AZ = np.matmul(A_dense, Z)
    Z = Z * z_scale
    AZ = AZ * z_scale

    Z_0 = (1.0 / (lambda2 + lambda3)) * (lambda3 * Z + lambda2 * AZ)

    Time_step_num = len(feature_list)

    # variables in all time steps
    experiment_num = 10  # !!!10
    Z_all = [[] for i in range(experiment_num)]  # Z across different time steps
    X_all = [[] for i in range(experiment_num)]
    Treat_all = [[] for i in range(experiment_num)]
    Y1_all = [[] for i in range(experiment_num)]
    Y0_all = [[] for i in range(experiment_num)]
    ATEs_all = [[] for i in range(experiment_num)]

    dz = Z.shape[1]  # dim of Z

    centroid1_idx_list = [random.randint(0, X.shape[0] - 1) for i in range(experiment_num)]  # experiment_num
    Alpha = np.array([np.random.normal(1.0 - float(r) / P, 1.0 / P, dz) for r in range(P)])  # P x dZ
    # Alpha = row_norm(Alpha.T).T
    Beta = np.array([np.random.normal(0, 1, dz)])  # size= 1 x dz

    # average ate, std
    ave_ate = np.zeros(experiment_num)
    ave_std_ate = np.zeros(experiment_num)

    # simple mode
    for t in range(0, Time_step_num):
        X = feature_list[t]
        A_dense = network_list[t]
        if row_norm_A:
            A_dense = row_norm(A_dense)

        for exp_id in range(experiment_num):
            # simulate treatment, outcome, causal effect
            centroid1_idx = centroid1_idx_list[exp_id]
            # Z_c1 = Z[centroid1_idx, :]
            # Z_c0 = np.mean(Z, axis=0)
            Z_c1 = Z_0[centroid1_idx, :]  # time = 0
            Z_c0 = np.mean(Z_0, axis=0)

            if t > 0:
                # z_1 (history)
                Z1 = np.zeros((n, dz))
                treat_last = Treat_all[exp_id][-1]
                treat_last = treat_last.reshape((-1, 1))
                Z_past = Z_all[exp_id][-P:]

                for r in range(len(Z_past)):
                    Z_r = Z_past[-(r + 1)]  # n x dz
                    Z1 += np.multiply(Z_r, Alpha[r])

                # if t < P:
                #     Alpha_sum = np.sum(Alpha[:len(Z_past)], axis=0)
                #     Z1 /= Alpha_sum

                # closer to the center, decided by last step treatment
                Z1 += np.matmul(treat_last, Beta)

            # X_t
            # lda = LatentDirichletAllocation(n_components=50)
            # lda.fit(X)
            # Z3 = lda.transform(X)  # node x n_topics
            Z3 = X
            Z3 = Z3 * z_scale

            # Neighbor
            AZ = np.matmul(A_dense, Z3)  # neighbor propagation
            if row_norm_AZ:
                AZ = row_norm(AZ)

            if t > 0:
                Z1_mean = np.mean(Z1)
                Z2_mean = np.mean(AZ)
                Z3_mean = np.mean(Z3)
                X_mean = np.mean(X)
                print("Z1 mean: ", Z1_mean, ' Z2 mean: ', Z2_mean, " Z3_mean: ", Z3_mean, " X_mean: ", X_mean)
            if t == 0:
                Z2_mean = np.mean(AZ)
                Z3_mean = np.mean(Z3)
                X_mean = np.mean(X)
                print(' Z2 mean: ', Z2_mean, " Z3_mean: ", Z3_mean, " X_mean:", X_mean)

            # scale
            if t == 0:
                Z = (1.0 / (lambda2 + lambda3)) * (lambda3 * Z + lambda2 * AZ)
            else:
                Z = 1.0 / (lambda1 + lambda2 + lambda3) * (lambda1 * Z1 + lambda2 * AZ + lambda3 * Z3)

            elipson = np.random.normal(0, 0.001, dz)
            Z += elipson

            # standardize z ?
            if norm_z:
                MZ = np.mean(Z, axis=0)
                VZ = np.std(Z, axis=0)

                Z_mean_orin = np.mean(Z, axis=0)
                Z_std_orin = np.std(Z, axis=0)

                Z = (Z - Z_mean_orin) / Z_std_orin

            Z_all[exp_id].append(Z)
            X_all[exp_id].append(X)

            # original way
            '''
            if kernel == "linear":
                # precompute the similarity between each instance and the two centroids
                ZZ_c1 = np.matmul(Z, Z_c1)
                ZZ_c0 = np.matmul(Z, Z_c0)

            elif kernel == "rbf":
                ZZ_c1 = rbf_kernel(Z - Z_c1)  # size = node (distance)
                ZZ_c0 = rbf_kernel(Z - Z_c0)

            elif kernel == "poly":
                # second order
                ZZ_c1 = (np.matmul(Z, Z_c1) + B) ** 2
                ZZ_c0 = (np.matmul(Z, Z_c0) + B) ** 2

            p1 = ZZ_c1
            p0 = ZZ_c0

            propensity = np.divide(np.exp(p1), np.exp(p1) + np.exp(p0))  # size = node

            ps = pd.Series(np.squeeze(propensity))
            '''

            # treatment: Propensity = ZW + B, W: dz x 1
            '''
            # W_p = np.random.random((dz, 1))'
            W_p = np.random.normal(0, 0.001, size=(dz, 1))
            # Bias = np.random.random(1)
            propensity = np.matmul(Z, W_p)  # + Bias
            propensity = 1.0 / (1 + np.exp(-propensity))

            T = np.random.binomial(1, p=propensity)
            T = T.reshape(-1)
            '''
            T = real_treat_list[t]

            # outcome

            eta = np.random.normal(0, 1, X.shape[0]).reshape((-1, 1))  # sample noise from Gaussian

            W_z = np.random.random((dz, 1))
            Bias = np.random.random(1)

            Y1 = kappa1 * np.matmul(Z, W_z) + Bias + eta
            Y0 = np.matmul(Z, W_z) + Bias + eta

            # original way
            '''
            eta = np.random.normal(0, 1, X.shape[0]).reshape(-1)  # sample noise from Gaussian
            Y1 = kappa1 * (p1 + p0) + eta
            Y0 = kappa1 * (p0) + eta
            '''
            Y1 = Y1.reshape(-1)
            Y0 = Y0.reshape(-1)

            ATE = Y1 - Y0

            ave_ate[exp_id] += np.mean(Y1 - Y0)
            ave_std_ate[exp_id] += np.std(Y1 - Y0)

            print("t=", t, " exp_id=", exp_id, " z mean=", np.mean(Z), " ", np.std(Z), "T:",
                  np.count_nonzero(T) / T.size, "Y1:", np.mean(Y1), " ", np.std(Y1), "Y0:", np.mean(Y0), " ",
                  np.std(Y0),
                  " ATE:", np.mean(Y1 - Y0), np.std(Y1 - Y0))

            if saving:
                Z_ = tsn(n_components=2).fit_transform(Z)  # N X 2
                labels = T  # use treatment as the binary label
                treated_idx = np.where(T == 1)[0]
                controled_idx = np.where(T == 0)[0]
                fig3, ax3 = plt.subplots()
                ax3.scatter(Z_[treated_idx, 0], Z_[treated_idx, 1], 3, marker='o', color='red')
                ax3.scatter(Z_[controled_idx, 0], Z_[controled_idx, 1], 3, marker='o', color='blue')
                # ax1.scatter(Z_[controled_idx, 0], Z_[controled_idx, 1], 3,marker='o',color='yellow')
                ax3.scatter(np.mean(Z_[:, 0]), np.mean(Z_[:, 1]), 100, label=r'$z_0^c$', marker='D', color='yellow')
                # ax3.scatter(Z_[centroid1_idx, 0], Z_[centroid1_idx, 1], 100, label=r'$z_1^c$', marker='D',
                #            color='green')
                # fig2, ax2 = plt.subplots()

                # ax2.scatter(np.mean(Z_[:,0]),np.mean(Z_[:,1]),100,label='centroid_0',marker='o',color='black')
                # ax2.scatter(Z_[centroid1_idx,0],Z_[centroid1_idx,1],100,label='centroid_1',marker='o',color='blue')
                plt.savefig('./figs/' + name + extra_str + 'time' + str(t) + '_exp' + str(exp_id) + 'tsne.pdf',
                            bbox_inches='tight')
                plt.legend(loc=2)
                plt.xlim(-100, 100)
                # pylab.show()

                #
                fig0, ax0 = plt.subplots()
                ax0.hist(propensity, bins=50)
                plt.title('propensity score distribution')
                plt.xlabel('propensity score')
                plt.ylabel('frequency')
                plt.savefig('./figs/' + name + extra_str + 'time' + str(t) + '_exp' + str(exp_id) + 'ps_dist.pdf',
                            bbox_inches='tight')

                fig1, ax1 = plt.subplots()
                ax1.hist(Y1, bins=50, label='Treated')
                ax1.hist(Y0, bins=50, label='Control')
                plt.title('outcome distribution')
                plt.legend()
                plt.savefig('./figs/' + name + extra_str + 'time' + str(t) + '_exp' + str(exp_id) + 'outcome_dist.pdf',
                            bbox_inches='tight')

                fig2, ax2 = plt.subplots()
                ax2.hist(Y1 - Y0, bins=50, label='ITE')
                plt.title('ITE distribution')
                plt.xlabel('ITE')
                plt.ylabel('frequency')
                ax2.axvline(x=np.mean(Y1 - Y0), color='red', label='ATE')
                plt.savefig('./figs/' + name + extra_str + 'time' + str(t) + '_exp' + str(exp_id) + 'ite_dist.pdf',
                            bbox_inches='tight')
                ax2.legend()

            Treat_all[exp_id].append(T)
            Y1_all[exp_id].append(Y1)
            Y0_all[exp_id].append(Y0)
            ATEs_all[exp_id].append(ATE)

            #break  # only one
        print("time ", t, " finished!")

    ave_ate /= Time_step_num
    ave_std_ate /= Time_step_num
    print('averaged ate: ', ave_ate, ave_std_ate)

    # save the data
    if saving_all:
        for exp_id in range(experiment_num):
            trn_id_list = random.sample(range(n), int(n * trn_rate))
            not_trn = list(set(range(n)) - set(trn_id_list))
            tst_id_list = random.sample(not_trn, int(n * tst_rate))
            val_id_list = list(set(not_trn) - set(tst_id_list))
            trn_id_list.sort()
            val_id_list.sort()
            tst_id_list.sort()

            # compress to save -- feature_list_sparse: a list of csc_matrix
            feature_list_sparse = [csc_matrix(feat) for feat in feature_list]
            network_list_sparse = [csc_matrix(net) for net in network_list]

            scio.savemat(root_path + name + str(exp_id) + '.mat', {
                'X': feature_list_sparse, 'A': network_list_sparse,
                'Z': Z_all[exp_id], 'T': Treat_all[exp_id],
                'Y1': Y1_all[exp_id], 'Y0': Y0_all[exp_id],
                'trn_idx': trn_id_list, 'val_idx': val_id_list, 'tst_idx': tst_id_list
                # 'Z_init': Z_init
            })

    return Z_all, Treat_all, Y1_all, Y0_all, ATEs_all


def simulate_Flickr(root_path, feature_list, network_list):
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    # save to file
    save_dense = True
    if save_dense:
        scio.savemat(root_path + 'Flickr_feat_dense' + '.mat', {
            'X': feature_list  # do not record the init time stamp
        })
        print("save features done")

    # parameters
    kappa1 = 100
    kappa2 = 1

    B = 0.01
    z_scale = 0.01  # scale of z, to prevent overflow
    att = "something"
    kernel = "linear"  # linear or rbf
    saving = False
    saving_all = True
    row_norm_AZ = False
    row_norm_A = False
    norm_z = False

    name = 'Flickr'

    P = 3  # p-order
    lambda1 = 0.3  # history
    lambda2 = 0.3  # network
    lambda3 = 0.3  # x->z

    trn_rate = 0.6
    val_rate = 0.2
    tst_rate = 0.2

    if kappa2 != 0.1:
        extra_str = str(kappa2)
    else:
        extra_str = ''

    if att != "":
        if kernel == "linear":
            extra_str += 's'
        elif kernel == "rbf":
            extra_str += 'r'
        elif kernel == "poly":
            extra_str += 'n'  # stands for non-linear

    X = feature_list[0]
    A0 = network_list[0]
    A_dense = A0

    n = X.shape[0]
    dx = X.shape[1]

    if row_norm_A:
        A_dense = row_norm(A_dense)

    Z, AZ = initialization(X, A_dense, row_norm_AZ)
    # scale
    Z = Z * z_scale
    AZ = AZ * z_scale
    Z_0 = (1.0 / (lambda2 + lambda3)) * (lambda3 * Z + lambda2 * AZ)

    Time_step_num = len(feature_list)

    # variables in all time steps
    experiment_num = 10  #
    Z_all = [[] for i in range(experiment_num)]  # Z across different time steps
    X_all = [[] for i in range(experiment_num)]
    Treat_all = [[] for i in range(experiment_num)]
    Y1_all = [[] for i in range(experiment_num)]
    Y0_all = [[] for i in range(experiment_num)]
    ATEs_all = [[] for i in range(experiment_num)]

    dz = Z.shape[1]  # dim of Z

    centroid1_idx_list = [random.randint(0, X.shape[0] - 1) for i in range(experiment_num)]  # experiment_num
    centroid0_idx_list = [random.randint(0, X.shape[0] - 1) for i in range(experiment_num)]  # experiment_num
    Alpha = np.array([np.random.normal(1.0 - float(r) / P, 1.0 / P, dz) for r in range(P)])  # P x dZ
    Alpha = row_norm(Alpha.T).T
    Alpha *= 1.8   # scale
    Beta = np.array([np.random.normal(0, 0.02, dz)])  # size= 1 x dz

    # average ate, std
    ave_ate = np.zeros(experiment_num)
    ave_std_ate = np.zeros(experiment_num)

    # simple mode
    for t in range(0, Time_step_num):
        X = feature_list[t]
        A_dense = network_list[t]
        if row_norm_A:
            A_dense = row_norm(A_dense)

        for exp_id in range(experiment_num):
            # simulate treatment, outcome, causal effect
            # centroid1_idx = centroid1_idx_list[exp_id]
            # Z_c1 = Z_0[centroid1_idx, :]  # time = 0
            # centroid0_idx = centroid0_idx_list[exp_id]
            # Z_c0 = Z_0[centroid0_idx, :]
            # Z_c0 = np.mean(Z_0, axis=0)

            if t > 0:
                # z_1 (history)
                Z1 = np.zeros((n, dz))
                treat_last = Treat_all[exp_id][-1]
                treat_last = treat_last.reshape((-1, 1))
                Z_past = Z_all[exp_id][-P:]

                for r in range(len(Z_past)):
                    Z_r = Z_past[-(r + 1)]  # n x dz
                    Z1 += np.multiply(Z_r, Alpha[r])

                # normalization
                # if t < P:
                #    Alpha_sum = np.sum(Alpha[:len(Z_past)], axis=0)
                #    Z1 /= Alpha_sum

                # closer to the center, decided by last step treatment
                Z1 += np.matmul(treat_last, Beta)

            # X_t
            lda = LatentDirichletAllocation(n_components=50)
            lda.fit(X)
            Z3 = lda.transform(X)  # node x n_topics

            # scale z
            Z3 = Z3 * z_scale

            # Neighbor
            AZ = np.matmul(A_dense, Z3)  # neighbor propagation
            if row_norm_AZ:
                AZ = row_norm(AZ)

            # scale
            if t == 0:
                # Z = (1.0 / (lambda2 + lambda3)) * (lambda3 * Z + lambda2 * AZ)
                Z = Z_0
            else:
                Z = 1.0 / (lambda1 + lambda2 + lambda3) * (lambda1 * Z1 + lambda2 * AZ + lambda3 * Z3)

            elipson = np.random.normal(0, 0.001, dz)
            Z += elipson

            # standardize z
            if norm_z:
                MZ = np.mean(Z, axis=0)
                VZ = np.std(Z, axis=0)

                Z_mean_orin = np.mean(Z, axis=0)
                Z_std_orin = np.std(Z, axis=0)

                Z = (Z - Z_mean_orin) / Z_std_orin

            Z_all[exp_id].append(Z)
            X_all[exp_id].append(X)

            # original way of treatment

            t_rate = 0

            if t == 0:
                while (t_rate < 0.48 or t_rate > 0.49):
                    centroid1_idx = random.randint(0, X.shape[0] - 1)
                    Z_c1 = Z_0[centroid1_idx, :]  # time = 0
                    centroid0_idx = random.randint(0, X.shape[0] - 1)
                    Z_c0 = Z_0[centroid0_idx, :]

                    if kernel == "linear":
                        # precompute the similarity between each instance and the two centroids
                        ZZ_c1 = np.matmul(Z, Z_c1)
                        ZZ_c0 = np.matmul(Z, Z_c0)

                    elif kernel == "rbf":
                        ZZ_c1 = rbf_kernel(Z - Z_c1)  # size = node (distance)
                        ZZ_c0 = rbf_kernel(Z - Z_c0)

                    elif kernel == "poly":
                        # second order
                        ZZ_c1 = (np.matmul(Z, Z_c1) + B) ** 2
                        ZZ_c0 = (np.matmul(Z, Z_c0) + B) ** 2

                    p1 = ZZ_c1
                    p0 = ZZ_c0

                    propensity = np.divide(np.exp(p1), np.exp(p1) + np.exp(p0))  # size = node

                    ps = pd.Series(np.squeeze(propensity))

                    # treatment: Propensity = ZW + B, W: dz x 1
                    '''
                    #W_p = np.random.random((dz, 1))'
                    W_p = np.random.normal(0, 0.001, size=(dz, 1))
                    #Bias = np.random.random(1)
                    propensity = np.matmul(Z, W_p) #+ Bias
                    propensity = 1.0 / (1 + np.exp(-propensity))
                    '''
                    T = np.random.binomial(1, p=propensity)
                    T = T.reshape(-1)

                    t_rate = np.count_nonzero(T) / T.size

                centroid1_idx_list[exp_id] = centroid1_idx
                centroid0_idx_list[exp_id] = centroid0_idx

                print('0 idx:', centroid0_idx, ' 1 idx:', centroid1_idx)
            else:
                centroid1_idx = centroid1_idx_list[exp_id]
                Z_c1 = Z_0[centroid1_idx, :]  # time = 0
                centroid0_idx = centroid0_idx_list[exp_id]
                Z_c0 = Z_0[centroid0_idx, :]

                if kernel == "linear":
                    # precompute the similarity between each instance and the two centroids
                    ZZ_c1 = np.matmul(Z, Z_c1)
                    ZZ_c0 = np.matmul(Z, Z_c0)

                elif kernel == "rbf":
                    ZZ_c1 = rbf_kernel(Z - Z_c1)  # size = node (distance)
                    ZZ_c0 = rbf_kernel(Z - Z_c0)

                elif kernel == "poly":
                    # second order
                    ZZ_c1 = (np.matmul(Z, Z_c1) + B) ** 2
                    ZZ_c0 = (np.matmul(Z, Z_c0) + B) ** 2

                p1 = ZZ_c1
                p0 = ZZ_c0

                propensity = np.divide(np.exp(p1), np.exp(p1) + np.exp(p0))  # size = node

                ps = pd.Series(np.squeeze(propensity))

                # treatment: Propensity = ZW + B, W: dz x 1
                '''
                #W_p = np.random.random((dz, 1))'
                W_p = np.random.normal(0, 0.001, size=(dz, 1))
                #Bias = np.random.random(1)
                propensity = np.matmul(Z, W_p) #+ Bias
                propensity = 1.0 / (1 + np.exp(-propensity))
                '''
                T = np.random.binomial(1, p=propensity)
                T = T.reshape(-1)

            # outcome

            eta = np.random.normal(0, 1, X.shape[0]).reshape((-1, 1))  # sample noise from Gaussian

            W_z = np.random.random((dz, 1))
            Bias = np.random.random(1)

            Y1 = kappa1 * np.matmul(Z, W_z) + Bias + eta
            Y0 = np.matmul(Z, W_z) + Bias + eta

            # original way
            '''
            eta = np.random.normal(0, 1, X.shape[0]).reshape(-1)  # sample noise from Gaussian
            Y1 = kappa1 * (p1 + p0) + eta
            Y0 = kappa1 * (p0) + eta
            '''
            Y1 = Y1.reshape(-1)
            Y0 = Y0.reshape(-1)

            ATE = Y1 - Y0

            ave_ate[exp_id] += np.mean(Y1 - Y0)
            ave_std_ate[exp_id] += np.std(Y1 - Y0)

            print("t=", t, " exp_id=", exp_id, " z mean=", np.mean(Z), " ", np.std(Z), "T:",
                  np.count_nonzero(T) / T.size, "Y1:", np.mean(Y1), " ", np.std(Y1), "Y0:", np.mean(Y0), " ",
                  np.std(Y0),
                  " ATE:", np.mean(Y1 - Y0), np.std(Y1 - Y0))

            if saving:
                Z_ = tsn(n_components=2).fit_transform(Z)  # N X 2
                labels = T  # use treatment as the binary label
                treated_idx = np.where(T == 1)[0]
                controled_idx = np.where(T == 0)[0]
                fig3, ax3 = plt.subplots()
                ax3.scatter(Z_[treated_idx, 0], Z_[treated_idx, 1], 3, marker='o', color='red')
                ax3.scatter(Z_[controled_idx, 0], Z_[controled_idx, 1], 3, marker='o', color='blue')
                # ax1.scatter(Z_[controled_idx, 0], Z_[controled_idx, 1], 3,marker='o',color='yellow')
                ax3.scatter(np.mean(Z_[:, 0]), np.mean(Z_[:, 1]), 100, label=r'$z_0^c$', marker='D', color='yellow')
                # ax3.scatter(Z_[centroid1_idx, 0], Z_[centroid1_idx, 1], 100, label=r'$z_1^c$', marker='D',
                #            color='green')
                # fig2, ax2 = plt.subplots()

                ax3.scatter(Z_[centroid0_idx, 0], Z_[centroid0_idx, 1], 100, label='centroid_0', marker='o',
                            color='black')
                ax3.scatter(Z_[centroid1_idx, 0], Z_[centroid1_idx, 1], 100, label='centroid_1', marker='o',
                            color='green')
                plt.savefig('./figs/' + name + extra_str + 'time' + str(t) + '_exp' + str(exp_id) + 'tsne.pdf',
                            bbox_inches='tight')
                plt.legend(loc=2)
                plt.xlim(-100, 100)
                # pylab.show()

                #
                fig0, ax0 = plt.subplots()
                ax0.hist(propensity, bins=50)
                plt.title('propensity score distribution')
                plt.xlabel('propensity score')
                plt.ylabel('frequency')
                plt.savefig('./figs/' + name + extra_str + 'time' + str(t) + '_exp' + str(exp_id) + 'ps_dist.pdf',
                            bbox_inches='tight')

                fig1, ax1 = plt.subplots()
                ax1.hist(Y1, bins=50, label='Treated')
                ax1.hist(Y0, bins=50, label='Control')
                plt.title('outcome distribution')
                plt.legend()
                plt.savefig('./figs/' + name + extra_str + 'time' + str(t) + '_exp' + str(exp_id) + 'outcome_dist.pdf',
                            bbox_inches='tight')

                fig2, ax2 = plt.subplots()
                ax2.hist(Y1 - Y0, bins=50, label='ITE')
                plt.title('ITE distribution')
                plt.xlabel('ITE')
                plt.ylabel('frequency')
                ax2.axvline(x=np.mean(Y1 - Y0), color='red', label='ATE')
                plt.savefig('./figs/' + name + extra_str + 'time' + str(t) + '_exp' + str(exp_id) + 'ite_dist.pdf',
                            bbox_inches='tight')
                ax2.legend()

            Treat_all[exp_id].append(T)
            Y1_all[exp_id].append(Y1)
            Y0_all[exp_id].append(Y0)
            ATEs_all[exp_id].append(ATE)

            # break  # only one
        print("time ", t, " finished!")

    ave_ate /= Time_step_num
    ave_std_ate /= Time_step_num
    print('averaged ate: ', ave_ate, ave_std_ate)

    # save the data
    if saving_all:
        for exp_id in range(experiment_num):
            trn_id_list = random.sample(range(n), int(n * trn_rate))
            not_trn = list(set(range(n)) - set(trn_id_list))
            tst_id_list = random.sample(not_trn, int(n * tst_rate))
            val_id_list = list(set(not_trn) - set(tst_id_list))
            trn_id_list.sort()
            val_id_list.sort()
            tst_id_list.sort()

            # compress to save -- feature_list_sparse: a list of csc_matrix
            feature_list_sparse = [csc_matrix(feat) for feat in feature_list]
            network_list_sparse = [csc_matrix(net) for net in network_list]

            scio.savemat(root_path + name + str(exp_id) + '.mat', {
                'X': feature_list_sparse, 'A': network_list_sparse,
                'Z': Z_all[exp_id], 'T': Treat_all[exp_id],
                'Y1': Y1_all[exp_id], 'Y0': Y0_all[exp_id],
                'trn_idx': trn_id_list, 'val_idx': val_id_list, 'tst_idx': tst_id_list
                # 'Z_init': Z_init
            })

    return Z_all, Treat_all, Y1_all, Y0_all, ATEs_all


def print_statistics(feature_list, network_list):
    '''
    :param feature_list: list of node features, size = time step, each elem = n_t x d
    :param network_list: list of graph structure, size = time step, each elem =
    :return:
    '''
    time_step = len(feature_list)
    num_edges = [np.sum(network_list[t], axis=1) for t in range(time_step)]  # [time] x node
    num_nodes = [np.sum(num_edges[t]>0) for t in range(time_step)]  # [time]
    num_edges = [np.sum(network_list[t])/2 for t in range(time_step)]
    feat_dim = feature_list[0].shape[1]
    print('min node num: ', min(num_nodes), 'max node num: ', max(num_nodes), ' num of nodes:', num_nodes)
    print('min edge num: ', min(num_edges), 'max edge num: ', max(num_edges), ' num of edges:', num_edges)
    print('dim of feature: ', feat_dim)

    return

if __name__ == '__main__':
    dataName = 'Flickr'  # 'Flickr', 'BlogCatalog', 'PeerRead'
    if dataName == 'Flickr':
        path = '../dataset/Flickr/Flickr.mat'
        root_path = '../dataset/' + 'simulate_new/Flickr/'

        feature_list, network_list = data_load(dataName, path)  # get the time-series features and networks

        # print the statistics of datasets
        #print_statistics(feature_list, network_list)

        Z_all, Treat_all, Y1_all, Y0_all, ATEs_all = simulate_Flickr(root_path, feature_list, network_list)
    elif dataName == 'BlogCatalog':
        path = '../dataset/BlogCatalog/BlogCatalog0.mat'
        feature_list, network_list = data_load(dataName, path)  # get the time-series features and networks
        Z_all, Treat_all, Y1_all, Y0_all, ATEs_all = simulate_Flickr(root_path, feature_list, network_list)
    elif dataName == 'PeerRead':
        # preprocessing
        # path = '../dataset/peerRead/arxiv.cs.ai_2007-2017/train/' #'../dataset/peerRead/arxiv.cs.ai_2007-2017/train/'
        # feature_list, network_list = loadData_peer(path)
        path = '../dataset/simulate_new/PeerRead/peerRead.mat'
        Z_all, Treat_all, Y1_all, Y0_all, ATEs_all = simulate_peer(path)



