#!/usr/bin/env bash
gpu=0
python main.py --gpu_id $gpu --tgt c --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt c --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt c --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 2

python main.py --gpu_id $gpu --tgt n --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt n --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt n --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 2
#
python main.py --gpu_id $gpu --tgt s --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 0 &
python main.py --gpu_id $gpu --tgt s --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 1 &
python main.py --gpu_id $gpu --tgt s --batch 36 --std_weight 1e-6 --std_weight 1e-10 --name std --lr 0.01 --seed 2