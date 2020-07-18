#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# FedAvg CIFAR-10 T=3000 non-IID
python main.py --momentum 0.1 \
               --epochs 3000 \
               --seed 1 \
               --iid 2 \
               --beta 0 \
               --num_users 100 \
               --frac 0.1 \
               --dataset cifar10 \
               --loss_type fedavg \
               --lr_drop 0.9992 \
               --local_ep 1 \
               --lr 0.1 \
               --weight-decay 5e-4

# FedAvg CIFAR-10 T=3000 IID
python main.py --momentum 0.1 \
               --epochs 3000 \
               --seed 1 \
               --iid 0 \
               --beta 0 \
               --num_users 100 \
               --frac 0.1 \
               --dataset cifar10 \
               --loss_type fedavg \
               --lr_drop 0.9992 \
               --local_ep 1 \
               --lr 0.1 \
               --weight-decay 5e-4

# FedAvg CIFAR-10 T=600 non-IID
python main.py --momentum 0.1 \
               --epochs 600 \
               --seed 1 \
               --iid 2 \
               --beta 0 \
               --num_users 100 \
               --frac 0.1 \
               --dataset cifar10 \
               --loss_type fedavg \
               --lr_drop 0.996 \
               --local_ep 5 \
               --lr 0.1 \
               --weight-decay 5e-4

# FedProx CIFAR-10 T=3000 non-IID
python main.py --momentum 0.1 \
               --epochs 3000 \
               --seed 1 \
               --iid 2 \
               --beta 0 \
               --mu 1 \
               --num_users 100 \
               --frac 0.1 \
               --dataset cifar10 \
               --loss_type fedprox \
               --lr_drop 0.9992 \
               --local_ep 1 \
               --lr 0.1 \
               --weight-decay 5e-4

# FedMAX CIFAR-10 T=3000 non-IID
python main.py --momentum 0.1 \
               --epochs 3000 \
               --seed 1 \
               --iid 2 \
               --beta 1000 \
               --num_users 100 \
               --frac 0.1 \
               --dataset cifar10 \
               --loss_type fedmax \
               --lr_drop 0.9992 \
               --local_ep 1 \
               --lr 0.1 \
               --weight-decay 5e-4

