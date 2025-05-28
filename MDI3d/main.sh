#!/usr/bin/env bash
gpu=0
python main.py --gpu_id $gpu --tgt rc --std_w 1e-8 --svd_w 1e-5 --batch 36 --name rankncon --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt rc --std_w 1e-8 --svd_w 1e-5 --batch 36 --name rankncon --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt rc --std_w 1e-8 --svd_w 1e-5 --batch 36 --name rankncon --lr 0.01 --seed 2

python main.py --gpu_id $gpu --tgt rl --std_w 5e-8 --svd_w 1.5e-3 --batch 36 --name rankncon --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt rl --std_w 5e-8 --svd_w 1.5e-3 --batch 36 --name rankncon --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt rl --std_w 5e-8 --svd_w 1.5e-3 --batch 36 --name rankncon --lr 0.01 --seed 2

python main.py --gpu_id $gpu --tgt t --std_w 5e-8 --svd_w 8e-4 --batch 36 --name rankncon --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt t --std_w 5e-8 --svd_w 8e-4 --batch 36 --name rankncon --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt t --std_w 5e-8 --svd_w 8e-4 --batch 36 --name rankncon --lr 0.01 --seed 2
