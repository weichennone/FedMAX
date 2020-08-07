#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6


import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_size)
    
    for idx, (data, target) in enumerate(data_loader):
        
        data, target = data.cuda(), target.cuda()
        
        if args.dataset == 'emnist':
            target -= 1  # <<<emnist
        
        log_probs, _ = net_g(data)
        
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    
    return accuracy, test_loss


def test_syn_hard(net_g, testdata, testlabel):

    net_g.eval()
    # testing
    test_loss = 0
    for idx, (data, target) in enumerate(zip(testdata, testlabel)):
        log_probs, _ = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

    test_loss /= len(testdata)
    return test_loss

