gpu=1

CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset RCF_MNIST --data_dir /home/s4686009/remotedata/DGRegression/RCF_MNIST --mixtype random --name lp-rankncon  --batch_type 1 --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0 &
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset RCF_MNIST --data_dir /home/s4686009/remotedata/DGRegression/RCF_MNIST --mixtype random --name lp-rankncon  --batch_type 1 --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 1 &
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset RCF_MNIST --data_dir /home/s4686009/remotedata/DGRegression/RCF_MNIST --mixtype random --name lp-rankncon  --batch_type 1 --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 2


CUDA_VISIBLE_DEVICES=0 \
python main.py --dataset Dti_dg --data_dir /home/s4686009/remotedata/DGRegression/dti --mixtype kde  --name lp-rankncon  --kde_bandwidth 10.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0 &
CUDA_VISIBLE_DEVICES=0 \
python main.py --dataset Dti_dg --data_dir /home/s4686009/remotedata/DGRegression/dti --mixtype kde  --name lp-rankncon  --kde_bandwidth 10.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 1 &
CUDA_VISIBLE_DEVICES=0 \
python main.py --dataset Dti_dg --data_dir /home/s4686009/remotedata/DGRegression/dti --mixtype kde  --name lp-rankncon  --kde_bandwidth 10.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 2 
