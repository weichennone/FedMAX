#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import numpy as np
import torch
from torch import nn
from utils.options import args_parser
from models.Update import SynTrain
from models.Nets import MLP

from models.Fed import FedAvg
from models.test import test_syn_hard
import torch.backends.cudnn as cudnn


def adjust_learning_rate(lr, lr_drop):
    lr *= lr_drop
    return lr


def save_checkpoint(state, filename='trained.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    def cal_Bdiss(w_p, w_c):
        glob_g = 0
        for k1 in w_p:
            glob_g += torch.sum((w_p[k1] - w_c[k1])**2)
        return torch.sqrt(glob_g).item()

    # load dataset and split users
    if args.dataset == 'syn':
        alpha = 0
        beta = 0
        dict_data = {}
        dict_label = {}
        dict_model = {}
        dict_test = {}
        dict_test_l = {}
        udistribution = torch.distributions.normal.Normal(0, alpha)
        bdistribution = torch.distributions.normal.Normal(0, beta)
        genemodel = MLP()
        num_batch = 500 // args.num_users
        test_batch = 100 // args.num_users

        var = torch.eye(1024)
        for i in range(1, 1024):
            var[i][i] = i ** (-1.2)

        for i in range(args.num_users):
            print('user: {:d}'.format(i))
            u = udistribution.sample()
            b = bdistribution.sample()
            wdist = torch.distributions.normal.Normal(u, 1)
            vdist = torch.distributions.normal.Normal(b, 1)
            mean = vdist.sample(sample_shape=torch.Size([1024]))
            xdist = torch.distributions.multivariate_normal.MultivariateNormal(mean, var)
            genemodel.fc1.weight = torch.nn.Parameter(wdist.sample(sample_shape=torch.Size([512, 1024])))
            genemodel.fc1.bias = torch.nn.Parameter(wdist.sample(sample_shape=torch.Size([512])))
            genemodel.fc2.weight = torch.nn.Parameter(wdist.sample(sample_shape=torch.Size([10, 512])))
            genemodel.fc2.bias = torch.nn.Parameter(wdist.sample(sample_shape=torch.Size([10])))

            dict_data[i] = []
            dict_label[i] = []
            dict_model[i] = copy.deepcopy(genemodel.state_dict())
            for j in range(num_batch):
                samples = xdist.sample(sample_shape=torch.Size([args.local_bs]))
                dict_data[i].append(samples)
                labels = genemodel(samples)[0].clone().detach().argmax(dim=1)
                dict_label[i].append(labels)

            dict_test[i] = []
            dict_test_l[i] = []
            for j in range(test_batch):
                samples = xdist.sample(sample_shape=torch.Size([args.local_bs]))
                dict_test[i].append(samples)
                labels = genemodel(samples)[0].clone().detach().argmax(dim=1)
                dict_test_l[i].append(labels)

    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.dataset == 'syn':
        net_glob = MLP()
    else:
        exit('Error: unrecognized model')

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net_glob.parameters()])))
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    init_w_glob = copy.deepcopy(w_glob)
    # training
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    softmax = nn.Softmax()
    logsoftmax = nn.LogSoftmax()
    kldiv = nn.KLDivLoss(reduce=True)

    learning_rate = args.lr
    test_acc = []
    avg_loss = []
    test_loss = []
    similarities = []
    dissimilarities = []
    print('gamma: {:d}'.format(int(args.gamma)))

    # Original
    for iter in range(args.epochs):

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_locals, loss_locals = [], []
        act_set = []
        test_loss_local = []
        for i, idx in enumerate(idxs_users):
            # print('user: {:d}'.format(idx))
            local = SynTrain(args=args, samples=dict_data[idx], labels=dict_label[idx])
            w, loss, act = local.train(model=copy.deepcopy(net_glob), lr=learning_rate)

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            act_set.append(act)

            loss_test = test_syn_hard(copy.deepcopy(net_glob), dict_test[idx], dict_test_l[idx])
            test_loss_local.append(loss_test)

        # update global weights
        w_glob = FedAvg(w_locals)
        
        # calculate similarity
        similarity = 0
        for i, idx in enumerate(idxs_users):
            similarity += cal_Bdiss(copy.deepcopy(dict_model[idx]), copy.deepcopy(w_glob))
        similarity /= int(args.frac * args.num_users)
        similarities.append(similarity)

        # calculate dissimilarity
        acts = torch.stack(act_set, dim=0)
        act = torch.mean(acts, dim=0)
        dissimilarity = 0
        for i, idx in enumerate(idxs_users):
            dissimilarity += kldiv(logsoftmax(act), softmax(act_set[i]))
        dissimilarity /= int(args.frac * args.num_users)
        dissimilarities.append(dissimilarity)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_test_p = sum(test_loss_local) / len(test_loss_local)
        print('Round {:3d}, similarity {:.6f}, dissimilarity {:.6f}, '.format(iter, similarity, dissimilarity))

        avg_loss.append(loss_avg)
        test_loss.append(loss_test_p)

        learning_rate = adjust_learning_rate(learning_rate, args.lr_drop)

    np.savez('./accuracy-' + 'N00-' + str(args.epochs) + '-seed' + str(
        args.seed) + '-' + str(args.loss_type) + '-gamma' + str(args.gamma), 
             loss=np.array(avg_loss), 
             similarity=similarities, 
             dis=dissimilarities, 
             testloss=test_loss)

    learning_rate = args.lr
    test_acc = []
    test_loss = []
    avg_loss = []
    similarities = []
    dissimilarities = []
    args.loss_type = 'fedmax'
    args.gamma = 1000
    net_glob.load_state_dict(init_w_glob)
    print('gamma: {:d}'.format(int(args.gamma)))

    for iter in range(args.epochs):

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_locals, loss_locals = [], []
        act_set = []
        test_loss_local = []
        for i, idx in enumerate(idxs_users):
            
            local = SynTrain(args=args, samples=dict_data[idx], labels=dict_label[idx])
            w, loss, act = local.train(model=copy.deepcopy(net_glob), lr=learning_rate)

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            act_set.append(act)

            loss_test = test_syn_hard(copy.deepcopy(net_glob), dict_test[idx], dict_test_l[idx])
            test_loss_local.append(loss_test)

        # update global weights
        w_glob = FedAvg(w_locals)
        
        # calculate similarity
        similarity = 0
        for i, idx in enumerate(idxs_users):
            similarity += cal_Bdiss(copy.deepcopy(dict_model[idx]), copy.deepcopy(w_glob))
        similarity /= int(args.frac * args.num_users)
        similarities.append(similarity)

        # calculate dissimilarity
        acts = torch.stack(act_set, dim=0)
        act = torch.mean(acts, dim=0)
        dissimilarity = 0
        for i, idx in enumerate(idxs_users):
            dissimilarity += kldiv(logsoftmax(act), softmax(act_set[i]))
        dissimilarity /= int(args.frac * args.num_users)
        dissimilarities.append(dissimilarity)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_test_p = sum(test_loss_local) / len(test_loss_local)
        print('Round {:3d}, similarity {:.6f}, dissimilarity {:.6f}, '.format(iter, similarity, dissimilarity))

        avg_loss.append(loss_avg)
        test_loss.append(loss_test_p)

        learning_rate = adjust_learning_rate(learning_rate, args.lr_drop)

    np.savez('./accuracy-' + 'N00-' + str(args.epochs) + '-seed' + str(
        args.seed) + '-' + str(args.loss_type) + '-gamma' + str(args.gamma), loss=np.array(avg_loss),
             similarity=similarities, dis=dissimilarities, testloss=test_loss)


