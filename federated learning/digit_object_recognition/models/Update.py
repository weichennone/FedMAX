#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().cuda()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, model, lr):
        """Train for one epoch on the training set"""
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    nesterov=self.args.nesterov,
                                    weight_decay=self.args.weight_decay)

        # hyper-parameters
        beta = self.args.beta

        # switch to train mode
        w_glob = copy.deepcopy(model.state_dict())
        l2_norm = nn.MSELoss()
        model.train()

        def cal_uniform_act(out):
            zero_mat = torch.zeros(out.size()).cuda()
            softmax = nn.Softmax(dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            
            kldiv = nn.KLDivLoss(reduce=True)
            cost = beta * kldiv(logsoftmax(out), softmax(zero_mat))
            
            return cost

        def cal_uniform_act_l2(out):
            cost = beta * torch.norm(out)
            return cost

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for i, (input, target) in tqdm(enumerate(self.ldr_train), total=len(self.ldr_train)):
                
                if self.args.dataset == 'emnist':
                    target -= 1  # emnist
                
                target = target.cuda()
                input = input.cuda()
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # compute output
                output, act = model(input_var)
                loss = self.loss_func(output, target_var)
                batch_loss.append(loss.clone().detach().item())

                # extra cost
                if self.args.loss_type == 'fedprox':
                    reg_loss = 0
                    for name, param in model.named_parameters():
                        reg_loss += l2_norm(param, w_glob[name])
                    loss += self.args.mu / 2 * reg_loss
                elif self.args.loss_type == 'fedmax':
                    loss += cal_uniform_act(act)
                elif self.args.loss_type == 'l2':
                    loss += cal_uniform_act_l2(act)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss) 


class SynTrain(object):
    def __init__(self, args, samples, labels):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = samples
        self.labels = labels

    def train(self, model, lr):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    nesterov=self.args.nesterov,
                                    weight_decay=self.args.weight_decay)

        # hyper-parameters
        beta = self.args.beta

        # switch to train mode
        model.train()

        def cal_uniform_act(out):
            shape = out.size()
            zero_mat = torch.zeros(shape)
            softmax = nn.Softmax(dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            kldiv = nn.KLDivLoss(reduce=True)
            cost = beta * kldiv(logsoftmax(out), softmax(zero_mat))
            return cost

        epoch_loss = []
        acts = torch.Tensor()

        for iter in range(self.args.local_ep):
            for i, (input, target) in enumerate(zip(self.ldr_train, self.labels)):
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # compute output
                output, act = model(input_var)
                loss = self.loss_func(output, target_var)
                
                if acts.size() == torch.Size([0]):
                    acts = act.clone().detach()
                else:
                    acts = torch.cat((acts, act.clone().detach()))

                # extra cost
                if self.args.loss_type == 'fedmax':
                    loss += cal_uniform_act(act)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), torch.mean(acts, dim=0)

