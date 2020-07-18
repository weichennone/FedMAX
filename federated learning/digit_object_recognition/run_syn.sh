#!/usr/bin/env bash
python main_synthetic.py --model mlp \
                         --epochs 200 \
                         --seed 1 \
                         --beta 0 \
                         --num_users 10 \
                         --frac 1 \
                         --dataset syn \
                         --loss_type none \
                         --lr_drop 0.992 \
                         --local_ep 1 \
                         --lr 0.1 \
                         --weight-decay 0
