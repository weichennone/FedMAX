#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# FedAvg
python train.py --arch resnet50 \
                --epochs 300 \
                --train_dataset chestxray \
                --num_users 100 \
                --iid 0 \
                --seed 1 \
                --beta 0

python train.py --arch resnet50 \
                --epochs 300 \
                --num_users 100 \
                --iid 0 \
                --seed 1 \
                --beta 0

# FedMAX
python train.py --arch resnet50 \
                --epochs 300 \
                --train_dataset chestxray \
                --num_users 100 \
                --iid 0 \
                --seed 1 \
                --beta 10000

python train.py --arch resnet50 \
                --epochs 300 \
                --num_users 100 \
                --iid 0 \
                --seed 1 \
                --beta 1000