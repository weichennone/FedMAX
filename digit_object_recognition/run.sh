#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# FedAvg CIFAR-10 T=3000 non-IID
python main.py --epochs 3000 \
               --lr_drop 0.9992 \
               --local_ep 1 \
               --num_users 100 \
               --frac 0.1 \
               --seed 1 \
               --iid 2 \
               --dataset cifar10 \
               --loss_type fedavg \

# FedAvg CIFAR-10 T=3000 IID
python main.py --epochs 3000 \
               --lr_drop 0.9992 \
               --local_ep 1 \
               --num_users 100 \
               --frac 0.1 \
               --seed 1 \
               --iid 0 \
               --dataset cifar10 \
               --loss_type fedavg \

# FedAvg CIFAR-10 T=600 non-IID
python main.py --epochs 600 \
               --lr_drop 0.996 \
               --local_ep 5 \
               --num_users 100 \
               --frac 0.1 \
               --seed 1 \
               --iid 2 \
               --dataset cifar10 \
               --loss_type fedavg \

# FedAvg FEMNIST* T=600 non-IID
python main.py --epochs 600 \
               --lr_drop 0.996 \
               --local_ep 5 \
               --num_users 100 \
               --frac 0.1 \
               --seed 1 \
               --iid 2 \
               --dataset emnist \
               --loss_type fedavg \

# FedAvg CIFAR-100 T=600 non-IID
python main.py --epochs 600 \
               --lr_drop 0.996 \
               --local_ep 5 \
               --num_users 100 \
               --frac 0.1 \
               --seed 1 \
               --iid 2 \
               --dataset cifar100 \
               --loss_type fedavg \

# FedProx CIFAR-10 T=3000 non-IID
python main.py --epochs 3000 \
               --lr_drop 0.9992 \
               --local_ep 1 \
               --num_users 100 \
               --frac 0.1 \
               --seed 1 \
               --iid 2 \
               --dataset cifar10 \
               --loss_type fedprox \
               --mu 1 \

# FedMAX CIFAR-10 T=3000 non-IID
python main.py --epochs 3000 \
               --lr_drop 0.9992 \
               --local_ep 1 \
               --num_users 100 \
               --frac 0.1 \
               --seed 1 \
               --iid 2 \
               --dataset cifar10 \
               --loss_type fedmax \
               --beta 1000 \

