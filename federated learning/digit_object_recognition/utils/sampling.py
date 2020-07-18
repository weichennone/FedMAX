#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import copy


def femnist_star(dataset, num_users):
    """
        Sample non-IID client data from EMNIST dataset -> FEMNIST*
        :param dataset:
        :param num_users:
        :return: dict of image index
    """
    print("Sampling dataset: FEMNIST*")
    dict_users = {i: [] for i in range(num_users)}
    total_len = len(dataset)

    labels = dataset.targets.numpy()
    idxs = np.argsort(labels)

    num_shards, num_imgs = 26 * num_users, total_len // (num_users * 26)

    label_selected = [np.random.choice(26, 20, replace=False) for _ in range(num_users)]

    label_selected_1 = [np.random.choice(label_selected[i], 6, replace=False) for i in range(num_users)]
    for i in range(num_users):
        for j in label_selected[i]:
            ind_pos = np.random.choice(num_users)
            tmp = copy.deepcopy(idxs[j * num_users * num_imgs + ind_pos * num_imgs: j * num_users * num_imgs + (ind_pos + 1) * num_imgs])
            dict_users[i].append(tmp)
        for j in label_selected_1[i]:
            ind_pos = np.random.choice(num_users)
            tmp = copy.deepcopy(idxs[j * num_users * num_imgs + ind_pos * num_imgs: j * num_users * num_imgs + (
                        ind_pos + 1) * num_imgs])
            dict_users[i].append(tmp)

    for i in range(num_users):
        dict_users[i] = np.concatenate(tuple(dict_users[i]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
        Sample IID client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
    """
    print("Sampling dataset: CIFAR-10 IID")
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid_2(dataset, num_users):
    """
        Sample non-IID client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
    """
    print("Sampling dataset: CIFAR-10 non-IID")
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    total_len = len(dataset)
    num_shards, num_imgs = 2 * num_users, 25000 // num_users
    idx_shard = [i for i in range(num_shards)]
    idxs = np.arange(total_len)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_100_noniid(dataset, num_users):
    """
        Sample non-IID client data from CIFAR100 dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
    """
    print("Sampling dataset: CIFAR-100 non-IID")
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    total_len = len(dataset)
    num_shards, num_imgs = 20 * num_users, total_len // (num_users * 20)

    labels = np.array(dataset.targets)
    idxs = np.argsort(labels)
    idx_shard = [i for i in range(num_shards)]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 20, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


def cifar_100_iid(dataset, num_users):
    """
        Sample IID client data from CIFAR100 dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
    """
    print("Sampling dataset: CIFAR-100 IID")
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

