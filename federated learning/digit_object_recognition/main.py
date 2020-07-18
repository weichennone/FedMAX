#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import cifar_iid, cifar_noniid_2, femnist_star, cifar_100_noniid, cifar_100_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNCifarStd5, CNNEmnistStd5, CNNCifar100Std5

from models.Fed import FedAvg
from models.test import test_img
import torch.backends.cudnn as cudnn


def adjust_learning_rate(lr, lr_drop):
    lr *= lr_drop
    return lr


def save_checkpoint(state, filename='trained.tar'):
    torch.save(state, filename)


def save_result(test_acc, avg_loss, filename=''):
    np.savez(filename, acc=np.array(test_acc), loss=np.array(avg_loss))


def main():
    # parse args
    args = args_parser()

    # random seed
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'cifar10':
        _CIFAR_TRAIN_TRANSFORMS = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        dataset_train = datasets.CIFAR10(
            './datasets/cifar10', train=True, download=True,
            transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS))

        _CIFAR_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        dataset_test = datasets.CIFAR10(
            './datasets/cifar10', train=False,
            transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS))

        if args.iid == 0:  # IID
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.iid == 2:  # non-IID
            dict_users = cifar_noniid_2(dataset_train, args.num_users)
        else:
            exit('Error: unrecognized class')
    
    elif args.dataset == 'emnist':
        _MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        dataset_train = datasets.EMNIST(
            './datasets/emnist', train=True, download=True,
            transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS),
            split='letters'
        )
        dataset_test = datasets.EMNIST(
            './datasets/emnist', train=False, download=True,
            transform=transforms.Compose(_MNIST_TEST_TRANSFORMS),
            split='letters'
        )

        dict_users = femnist_star(dataset_train, args.num_users)
    
    elif args.dataset == 'cifar100':
        _CIFAR_TRAIN_TRANSFORMS = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        dataset_train = datasets.CIFAR100(
            './datasets/cifar100', train=True, download=True,
            transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS))

        _CIFAR_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        dataset_test = datasets.CIFAR100(
            './datasets/cifar100', train=False,
            transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS))
        if args.iid == 0:  # IID
            dict_users = cifar_100_iid(dataset_train, args.num_users)
        elif args.iid == 2:  # non-IID
            dict_users = cifar_100_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.dataset == 'cifar10':
        if args.model == "CNNStd5":
            net_glob = CNNCifarStd5().cuda()
        else:
            exit('Error: unrecognized model')
    elif args.dataset == 'emnist':
        if args.model == "CNNStd5":
            net_glob = CNNEmnistStd5().cuda()
        else:
            exit('Error: unrecognized model')
    elif args.dataset == 'cifar100':
        if args.model == "CNNStd5":
            net_glob = CNNCifar100Std5().cuda()
        else:
            exit('Error: unrecognized model')
    else:
        exit('Error: unrecognized model')

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net_glob.parameters()])))
    
    net_glob.train()

    learning_rate = args.lr
    test_acc = []
    avg_loss = []

    # Train
    for iter in range(args.epochs):

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_locals, loss_locals = [], []
        for i, idx in enumerate(idxs_users):
            print('user: {:d}'.format(idx))
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(model=copy.deepcopy(net_glob).cuda(), lr=learning_rate)

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.6f}'.format(iter, loss_avg))

        acc_test, _ = test_img(net_glob.cuda(), dataset_test, args)
        print("test accuracy: {:.4f}".format(acc_test))
        test_acc.append(acc_test)

        avg_loss.append(loss_avg)

        learning_rate = adjust_learning_rate(learning_rate, args.lr_drop)

    filename = './accuracy-' + str(args.dataset) + '-iid' + str(args.iid) + '-' + str(args.epochs) + '-seed' \
               + str(args.seed) + '-' + str(args.loss_type) + '-gamma' + str(args.gamma) + '-mu' + str(args.mu)
    save_result(test_acc, avg_loss, filename)
    

if __name__ == '__main__':
    main()
