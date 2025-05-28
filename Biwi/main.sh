#!/usr/bin/env bash
gpu=0
python main.py --gpu_id $gpu --tgt m --batch 18 --svd_weight 0.5 --std_weight 1e-7 --name rankncon --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt m --batch 18 --svd_weight 0.5 --std_weight 1e-7 --name rankncon --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt m --batch 18 --svd_weight 0.5 --std_weight 1e-7 --name rankncon --lr 0.01 --seed 2

python main.py --gpu_id $gpu --tgt f --batch 18 --svd_weight 0.5 --std_weight 1e-7 --name rankncon --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt f --batch 18 --svd_weight 0.5 --std_weight 1e-7 --name rankncon --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt f --batch 18 --svd_weight 0.5 --std_weight 1e-7 --name rankncon --lr 0.01 --seed 2