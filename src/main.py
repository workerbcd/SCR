# ContinuousMIXUP
import shutil

from data_generate import load_data
from utils import set_seed, stats_values, get_unique_file_name, write_result, write_model, timestamp
from config import dataset_defaults

from tokenize import Special
import algorithm
from models import Learner, Learner_TimeSeries,Learner_Dti_dg, Learner_RCF_MNIST
from torchvision import models
import copy
import numpy as np
import torch
import argparse
import random

import pickle
# import ipdb
import os
import matplotlib.pyplot as plt
import time
import io
import torch.nn as nn


############ cmd process ##############
parser = argparse.ArgumentParser(description='kde + mixup')
parser.add_argument('--result_root_path', type = str, default="/home/s4686009/remotedata/DGRegression/logs/",
                    help="path to store the results")
parser.add_argument('--dataset', type=str, default='NO2', 
                    help='dataset')
parser.add_argument('--mixtype', type=str, default='random',
                    help="random or kde or erm")
parser.add_argument('--use_manifold', type=int, default=1,
                    help='use manifold mixup or not')
parser.add_argument('--seed', type=int, default=0,
                    help="seed")
parser.add_argument('--gpu', type=int, default=0,
                    help="train on which cuda device")

#### kde parameter ####
parser.add_argument('--kde_bandwidth', type=float, default=1.0,
                    help="bandwidth")
parser.add_argument('--kde_type', type=str, default='gaussian', help = 'gaussian or tophat')
parser.add_argument('--batch_type', default=0, type=int, help='1 for y batch and 2 for x batch and 3 for representation')

#### verbose ####
parser.add_argument('--show_process', type=int, default = 1,
                    help = 'show rmse and r^2 in the process')
parser.add_argument('--show_setting', type=int, default = 1,
                    help = 'show setting')

#### model read & write ####
parser.add_argument('--read_best_model', type=int, default=0, help='read from original model')
parser.add_argument('--store_model', type=int, default=1, 
                    help = 'store model or not')

########## data path, for RCF_MNIST and TimeSeries #########
parser.add_argument('--data_dir', type = str, help = 'for RCF_MNIST and TimeSeries')

parser.add_argument('--ts_name', type=str,  default='',
                    help='ts dataset name')
parser.add_argument('--name', type=str, default='', help='restore name')
#########hyperparameter##########
parser.add_argument('--std_w', type=float, default=0.0, help="std weight")
parser.add_argument('--svd_w', type=float, default=0.0, help="svd weight")
parser.add_argument('--uni_batchsize', type=int, default=0, help="universal batch size")

parser.add_argument('--cut_rate', type=float,default=1.0)
########## cmd end ############

args = parser.parse_args()
args.cuda = torch.cuda.is_available() # for ts_data init function
args_dict = args.__dict__
dict_name = args.dataset
if args.dataset == 'TimeSeries':
    dict_name += '-' + args.ts_name
args_dict.update(dataset_defaults[dict_name])
# if args.std_w > 0:
#     args.std_weight = args.std_w
#
# if args.svd_w > 0:
#     args.svd_weight = args.svd_w
if args.uni_batchsize !=0:
    args.batch_size = args.uni_batchsize

args = argparse.Namespace(**args_dict)
if args.show_setting: # basic information
    for k in dataset_defaults[dict_name].keys():
        print(f'{k}: {dataset_defaults[dict_name][k]}')

########## device ##########

if torch.cuda.is_available() and args.gpu != -1:
    torch.cuda.set_device('cuda:'+str(args.gpu))
    device = torch.device('cuda:'+str(args.gpu))
    if args.show_setting:
        print(device)
else:
    device = torch.device('cpu')
    if args.show_setting:
        print("use cpu")

set_seed(args.seed) # init set

####### mkdir result path ########
# result_root = os.path.join(args.result_root_path, 'ablation/')
result_root = args.result_root_path
if not os.path.exists(result_root):
    os.makedirs(result_root)
print(args.name)
stamp = timestamp()
result_path = result_root + f"{args.dataset}/{stamp}_{args.name}_svdw{args.svd_weight}_stdw{args.std_weight}_batchsize{args.batch_size}_cutrate{args.cut_rate}_seed{args.seed}/"
if args.ts_name!='':
    result_path = result_root + f"{args.dataset}/{args.ts_name}/{stamp}_{args.name}_svdw{args.svd_weight}_stdw{args.std_weight}_seed{args.seed}/"

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(result_path)
shutil.copytree('.', os.path.join(result_path, 'src') )



def load_model(args, ts_data):
    if args.dataset == 'TimeSeries':
        model = Learner_TimeSeries(args=args,data=ts_data).to(device)
    elif args.dataset == 'Dti_dg':
        model = Learner_Dti_dg(hparams=None).to(device)
    elif args.dataset == 'RCF_MNIST':
        model = Learner_RCF_MNIST(args=args).to(device)
    else:
        model = Learner(args=args).to(device)
    
    if args.show_setting:
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of parameters: %d' % nParams)
    return model


def main():
    t1 = time.time()
    best_model_dict = {}
    data_packet, ts_data = load_data(args)
    if args.show_setting:
        print('load dataset success, use time = {:.4f}'.format(time.time() - t1))
        print(f'args.mixtype = {args.mixtype}, Use_manifold = {args.use_manifold}')
    
    set_seed(args.seed) # seed aligned 

    if args.read_best_model == 0: # normal train
        #### model ####
        model = load_model(args,ts_data)
        if args.show_setting:
            print('load untrained model done')
            print(args)
        
        all_begin = time.time()

        #### get mixup sample rate among data ####
        print('-------------------------------------',args.mixtype)
        if args.uni_batchsize != 0:
            args.batch_size = args.uni_batchsize
        print("batch size is {}".format(args.batch_size))
        print(args.uni_batchsize)
        if ('cmixup' in args.mixtype) or (args.mixtype=='kde'):
            mixup_idx_sample_rate = algorithm.get_mixup_sample_rate(args, data_packet, device,use_kde=True)
        else:
            mixup_idx_sample_rate = None
        
        sample_use_time = time.time() - all_begin
        print('sample use time = {:.4f}'.format(sample_use_time))

        #### train model ####
        best_model_dict['rmse'], best_model_dict['r'] = \
            algorithm.train(args, model, data_packet, args.mixtype == "kde" or args.mixtype=="random",\
                            mixup_idx_sample_rate,ts_data, device,result_path=result_path)
        
        print('='*30 + ' single experiment result ' + '=' * 30)
        y_train = data_packet['y_train']
        if isinstance(y_train,torch.FloatTensor):
            y_train = y_train.cpu().numpy()
        if args.label_factor:
            y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train,axis=0)
        else:
            y_mean,y_std=0.0, 1.0
        result_dict_best = algorithm.test(args, best_model_dict[args.metrics], data_packet['x_test'], data_packet['y_test'],
                                        'seed = ' + str(args.seed) + ': Final test for best ' + args.metrics + ' model: ' + args.mixtype + ', use_manifold = ' + str(args.use_manifold) + ', kde_bandwidth = ' + str(args.kde_bandwidth) + ':\n',
                                        args.show_process, all_begin, device,y_train_mean=y_mean, y_train_std=y_std)
        
        algorithm.cal_worst_acc(args,data_packet,best_model_dict[args.metrics],
                                result_dict_best, all_begin,ts_data,device,y_train_mean=y_mean, y_train_std=y_std)

        # write results
        write_result(args, args.kde_bandwidth, result_dict_best, result_path)
        if args.store_model:
            write_model(args, best_model_dict[args.metrics], result_path)

    else: # use best model, 1 for rmse or 2 for r
        assert args.read_best_model == 1
        # extra_str = '' if args.metrics == 'rmse' else 'r'
        pt_full_path = result_path + get_unique_file_name(args, '','.pickle')
        
        with open(pt_full_path,'rb') as f:
            s = f.read()
            read_model = pickle.loads(s)
        print('load best model success from {pt_full_path}!')

        all_begin = time.time()
        
        print('='*30 + ' read best model and verify result ' + '=' * 30)
        read_result_dic = algorithm.test(args, read_model, data_packet['x_test'], data_packet['y_test'],
                        ('seed = ' + str(args.seed) + ': Final test for read model: ' + pt_full_path + ':\n'),
                        True, all_begin,  device)            
                        
        algorithm.cal_worst_acc(args,data_packet,read_model,read_result_dic,all_begin,ts_data, device)
        
        write_result(args, 'read', read_result_dic, result_path, '') # rewrite result txt

if __name__ == '__main__':
    main()