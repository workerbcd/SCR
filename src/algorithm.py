import os.path

import numpy as np
import copy
import math
# import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim import Adam, SGD
from sklearn.neighbors import KernelDensity
from utils import stats_values
from preprocess import getDomainPairs,get_domain_pairs, get_label_pairs
from models import Discriminator
from losses import DomainTripletLoss, ContrastiveRegression,\
    GaussianApproximation,BasisAlignment,adv_loss, RegMetricLoss, weighted_mse_loss
from loss.ranksim import batchwise_ranking_regularizer
from gradient_align import fish_step
from postprocess import visualization, load_model, save_model
from lds import _prepare_weights


def cal_worst_acc(args,data_packet,best_model_rmse,best_result_dict_rmse,all_begin,ts_data,device,y_train_mean=0,y_train_std=1.0):
    #### worst group acc ---> rmse ####
    if args.is_ood:
        x_test_assay_list = data_packet['x_test_assay_list']
        y_test_assay_list = data_packet['y_test_assay_list']
        worst_acc = 0.0 if args.metrics == 'rmse' else 1e10
            
        for i in range(len(x_test_assay_list)):
            result_dic = test(args,best_model_rmse,x_test_assay_list[i],y_test_assay_list[i],
                            '', False, all_begin, device,y_train_mean=y_train_mean,y_train_std=y_train_std)
            acc = result_dic[args.metrics] 
            if args.metrics == 'rmse':
                if acc > worst_acc:
                    worst_acc = acc
            else:#r
                if np.abs(acc) < np.abs(worst_acc):
                    worst_acc = acc
        print('worst {} = {:.3f}'.format(args.metrics, worst_acc))
        best_result_dict_rmse['worst_' + args.metrics] = worst_acc

def get_mixup_sample_rate(args, data_packet, device='cuda', use_kde = False):
    
    mix_idx = []
    _, y_list = data_packet['x_train'], data_packet['y_train'] 
    is_np = isinstance(y_list,np.ndarray)
    if is_np:
        data_list = torch.tensor(y_list, dtype=torch.float32)
    else:
        data_list = y_list

    N = len(data_list)
    ######## use kde rate or uniform rate #######
    for i in range(N):
        if 'cmixup' in args.mixtype or use_kde: # kde
            data_i = data_list[i]
            data_i = data_i.reshape(-1,data_i.shape[0]) # get 2D
            
            if args.show_process:
                if i % (N // 10) == 0:
                    print('Mixup sample prepare {:.2f}%'.format(i * 100.0 / N ))
                # if i == 0: print(f'data_list.shape = {data_list.shape}, std(data_list) = {torch.std(data_list)}')#, data_i = {data_i}' + f'data_i.shape = {data_i.shape}')
                
            ######### get kde sample rate ##########
            kd = KernelDensity(kernel=args.kde_type, bandwidth=args.kde_bandwidth).fit(data_i)  # should be 2D
            each_rate = np.exp(kd.score_samples(data_list))
            each_rate /= np.sum(each_rate)  # norm
        else:
            each_rate = np.ones(y_list.shape[0]) * 1.0 / y_list.shape[0]
        
        ####### visualization: observe relative rate distribution shot #######
        if args.show_process and i == 0:
                print(f'bw = {args.kde_bandwidth}')
                print(f'each_rate[:10] = {each_rate[:10]}')
                stats_values(each_rate)
            
        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]

    if args. show_process:
        print(f'len(y_list) = {len(y_list)}, len(mix_idx) = {len(mix_idx)}, np.mean(self_rate) = {np.mean(self_rate)}, np.std(self_rate) = {np.std(self_rate)},  np.min(self_rate) = {np.min(self_rate)}, np.max(self_rate) = {np.max(self_rate)}')

    return mix_idx


def get_batch_kde_mixup_idx(args, Batch_X, Batch_Y, device):
    assert Batch_X.shape[0] % 2 == 0
    Batch_packet = {}
    Batch_packet['x_train'] = Batch_X.cpu()
    Batch_packet['y_train'] = Batch_Y.cpu()

    Batch_rate = get_mixup_sample_rate(args, Batch_packet, device, use_kde=True) # batch -> kde
    if args. show_process:
        stats_values(Batch_rate[0])
        # print(f'Batch_rate[0][:20] = {Batch_rate[0][:20]}')
    idx2 = [np.random.choice(np.arange(Batch_X.shape[0]), p=Batch_rate[sel_idx]) 
            for sel_idx in np.arange(Batch_X.shape[0]//2)]
    return idx2

def get_batch_kde_mixup_batch(args, Batch_X1, Batch_X2, Batch_Y1, Batch_Y2, device):
    Batch_X = torch.cat([Batch_X1, Batch_X2], dim = 0)
    Batch_Y = torch.cat([Batch_Y1, Batch_Y2], dim = 0)

    idx2 = get_batch_kde_mixup_idx(args,Batch_X,Batch_Y,device)

    New_Batch_X2 = Batch_X[idx2]
    New_Batch_Y2 = Batch_Y[idx2]
    return New_Batch_X2, New_Batch_Y2


def test(args, model, x_list, y_list, name, need_verbose, epoch_start_time, device,result_path=None,y_train_mean=0,y_train_std=1.0):
    model.eval()
    with torch.no_grad():
        if args.dataset == 'Dti_dg': 
            val_iter = x_list.shape[0] // args.batch_size 
            val_len = args.batch_size
            y_list = y_list[:val_iter * val_len]
        else: # read in the whole test data
            val_iter = 1
            val_len = x_list.shape[0]
        y_list_pred = []
        feat_list =[]
        assert val_iter >= 1 #  easy test

        for ith in range(val_iter):
            if isinstance(x_list,np.ndarray):
                x_list_torch = torch.tensor(x_list[ith*val_len:(ith+1)*val_len], dtype=torch.float32).to(device)
            else:
                x_list_torch = x_list[ith*val_len:(ith+1)*val_len].to(device)

            model = model.to(device)
            pred_y, feats = model(x_list_torch, return_feats=True)
            pred_y = pred_y.cpu().numpy()
            y_list_pred.append(pred_y)
            feat_list.append(feats.cpu().numpy())

        y_list_pred = np.concatenate(y_list_pred,axis=0)
        y_list = y_list.squeeze()
        # feats =
        y_list_pred = y_list_pred.squeeze()

        if not isinstance(y_list, np.ndarray):
            y_list = y_list.numpy()
        if args.label_factor:
            y_list_pred = y_list_pred*y_train_std + y_train_mean
            if len(y_list_pred.shape) > 1:
                y_list_pred = y_list_pred*y_list_pred.shape[-1]
        # print(y_list)
        # print(y_list_pred)
        ###### calculate metrics ######

        mean_p = y_list_pred.mean(axis = 0)
        sigma_p = y_list_pred.std(axis = 0)
        mean_g = y_list.mean(axis = 0)
        sigma_g = y_list.std(axis = 0)

        index = (sigma_g!=0)
        corr = ((y_list_pred - mean_p) * (y_list - mean_g)).mean(axis = 0) / (sigma_p * sigma_g)
        corr = (corr[index]).mean()

        mse = (np.square(y_list_pred  - y_list )).mean()
        result_dict = {'mse':mse, 'r':corr, 'r^2':corr**2, 'rmse':np.sqrt(mse)}

        not_zero_idx = y_list != 0.0
        mape = (np.fabs(y_list_pred[not_zero_idx] -  y_list[not_zero_idx]) / np.fabs(y_list[not_zero_idx])).mean() * 100
        result_dict['mape'] = mape
        
    ### verbose ###
    if need_verbose:
        epoch_use_time = time.time() - epoch_start_time
        # valid -> interval time; final test -> all time
        line = name + 'corr = {:.4f}, rmse = {:.4f}, mape = {:.4f} %'.format(corr, np.sqrt(mse),
                                                                            mape) + ', time = {:.4f} s'.format(
            epoch_use_time)
        print(line)
        if result_path is not None:
            file = os.path.join(result_path,'log.txt')
            with open(file, 'a+') as f:
                f.write(line+'\n')
    return result_dict


def train(args, model, data_packet, is_mixup=True, mixup_idx_sample_rate=None, ts_data= None, device='cuda',result_path=None):
    ######### model prepare ########
    model.train(True)
    model_inner = copy.deepcopy(model)
    # discriminator = Discriminator(model.n_feat).to(device)
    # optimizer_d = Adam(discriminator.parameters(), **args.optimiser_args)
    loss_fun = nn.MSELoss(reduction='mean').to(device)
    tri_loss = DomainTripletLoss(margin=args.margin)
    if result_path is not None:
        writer = SummaryWriter(os.path.join(result_path,'tensorboard'))
    root_path = args.result_root_path + f"{args.dataset}/basemodel/"
    if args.ts_name != '':
        root_path = args.result_root_path + f"{args.dataset}/{args.ts_name}/basemodel/"
    modeldir = os.path.join(root_path, f'seed{args.seed}')

    cr_loss = ContrastiveRegression(tau=2.0)

    best_mse = 1e10  # for best update
    best_r2 = 0.0
    repr_flag = 1 # for batch kde visualize training process

    scheduler = None

    x_train = data_packet['x_train']
    y_train = data_packet['y_train']
    x_valid = data_packet['x_valid']
    y_valid = data_packet['y_valid']
    train_size = len(x_train)
    x_train = x_train[:int(train_size*args.cut_rate)]
    y_train = y_train[:int(train_size*args.cut_rate)]
    print("x train size {}".format(x_train.shape))
    if isinstance(y_train, torch.FloatTensor):
        y_train = y_train.cpu().numpy()
    y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
    y_train_normlized = (y_train - y_mean) / y_std
    if len(y_train.shape) > 1:
        y_train_normlized = y_train_normlized / float(y_train.shape[-1])
    if args.label_factor:
        y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
        y_train = (y_train - y_mean) / y_std
        if len(y_train.shape) > 1:
            y_train = y_train / float(y_train.shape[-1])
    else:
        y_mean, y_std = 0.0, 1.0
    print("y_train size is {}".format(y_train.shape))

    gaussian_approximation = GaussianApproximation(model.n_feat,
                                                   y_train.shape[-1] if len(y_train.shape) > 1 else 1,
                                                   device, args)
    base_align = BasisAlignment(model.n_feat,args.batch_size,
                                y_train.shape[-1] if len(y_train.shape) > 1 else 1,device)

    iteration = len(x_train) // args.batch_size
    steps_per_epoch = iteration

    result_dict,best_mse_model = {},None
    step_print_num = 30 # for dti

    need_shuffle = not args.is_ood
    # need_shuffle=True
    # if 'contrast' in args.mixtype:
    # ind_same, ind_diff = getDomainPairs(zip(x_train,y_train),device,model,result_path)
    if 'contrast' in args.mixtype:
        ind_same, ind_diff = get_domain_pairs(args, x_train, y_train, device, model, path=result_path)
    else:
        ind_same, ind_diff = np.arange(len(x_train)),np.arange(len(x_train))
    optimizer = Adam([
        {'params':model.parameters(),'lr':args.optimiser_args['lr']},
        {'params':base_align.parameters(),'lr':args.optimiser_args['lr']*10}
        # {'params': gaussian_approximation.parameters(), 'lr': args.optimiser_args['lr'] * 10}
    ])
    optimizer_inner = Adam([{'params':model_inner.parameters(),'lr':args.optimiser_args['lr']}])
                            # {'params':gaussian_approximation.parameters(),'lr':args.optimiser_args['lr']*10}])
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.num_epochs*0.2),int(args.num_epochs*0.5)],gamma=0.1,last_epoch=-1)
    # if os.path.exists(modeldir):
    #     num_epochs=0
    # else:
    #     num_epochs=args.num_epochs
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        shuffle_idx = np.random.permutation(np.arange(len(x_train)))

        if need_shuffle: # id
            x_train_input = x_train[shuffle_idx]
            y_train_input = y_train[shuffle_idx]
            if 'contrast' in args.mixtype:
                ind_same_input = ind_same[shuffle_idx]
                ind_diff_input = ind_diff[shuffle_idx]
        else:# ood
            x_train_input = x_train
            y_train_input = y_train
            if 'contrast' in args.mixtype:
                ind_same_input = ind_same
                ind_diff_input = ind_diff
        if 'kde' in args.mixtype or 'random' in args.mixtype:  # mix up
            for idx in range(iteration):

                lambd = np.random.beta(args.mix_alpha, args.mix_alpha)

                if need_shuffle: # get batch idx
                    idx_1 = shuffle_idx[idx * args.batch_size:(idx + 1) * args.batch_size]
                else:
                    idx_1 = np.arange(len(x_train))[idx * args.batch_size:(idx + 1) * args.batch_size]

                if args.mixtype == 'kde':
                    idx_2 = np.array(
                        [np.random.choice(np.arange(x_train.shape[0]), p=mixup_idx_sample_rate[sel_idx]) for sel_idx in
                        idx_1])
                else: # random mix
                    idx_2 = np.array(
                        [np.random.choice(np.arange(x_train.shape[0])) for sel_idx in idx_1])

                if isinstance(x_train,np.ndarray):
                    X1 = torch.tensor(x_train[idx_1], dtype=torch.float32).to(device)
                    X2 = torch.tensor(x_train[idx_2], dtype=torch.float32).to(device)
                else:
                    X1 = x_train[idx_1].to(device)
                    X2 = x_train[idx_2].to(device)
                if isinstance(y_train, np.ndarray):
                    Y1 = torch.tensor(y_train[idx_1], dtype=torch.float32).to(device)
                    Y2 = torch.tensor(y_train[idx_2], dtype=torch.float32).to(device)
                else:
                    Y1 = y_train[idx_1].to(device)
                    Y2 = y_train[idx_2].to(device)

                if args.batch_type == 1: # sample from batch
                    assert args.mixtype == 'random'
                    if not repr_flag: # show the sample status once
                        args.show_process = 0
                    else:
                        repr_flag = 0
                    X2, Y2 = get_batch_kde_mixup_batch(args,X1,X2,Y1,Y2,device)
                    args.show_process = 1

                X1 = X1.to(device)
                X2 = X2.to(device)
                Y1 = Y1.to(device)
                Y2 = Y2.to(device)

                # mixup
                mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
                mixup_X = X1 * lambd + X2 * (1 - lambd)
                # forward
                if args.use_manifold == True:
                    pred_Y,feats_mixup = model.forward_mixup(X1, X2, lambd, return_feats=True)
                else:
                    pred_Y,feats_mixup = model.forward(mixup_X, return_feats=True)

                # pred, feats = model.forward(X1, return_feats=True)

                if args.dataset == 'TimeSeires': # time series loss need scale
                    scale = ts_data.scale.expand(pred_Y.size(0),ts_data.m)
                    loss_mse = loss_fun(pred_Y * scale, mixup_Y * scale)
                else:
                    loss_mse = loss_fun(pred_Y, mixup_Y)
                # loss_std, loss_ba = base_align.align_feat_bases(feats, feats_mixup, Y1, mixup_Y)
                # loss_std, loss_ba = base_align.align_feat_and_label(feats_mixup, mixup_Y)
                loss = loss_mse #+ 0.1*args.local_scale*loss_std

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.dataset == 'Dti_dg' and (idx-1) % (iteration // step_print_num) == 0: # dti has small epoch number, so verbose multiple times at 1 iteration
                    result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) + ',  step = {} '.format((epoch*steps_per_epoch + idx)) + ':\t', args.show_process, epoch_start_time, device)
                    # save best model
                    if result_dict['mse'] <= best_mse:
                        best_mse = result_dict['mse']
                        best_mse_model = copy.deepcopy(model)
                    if result_dict['r']**2 >= best_r2:
                        best_r2 = result_dict['r']**2
                        best_r2_model = copy.deepcopy(model)
        else:
            best_mse_model = model
            best_r2_model = model
            print('no pretrain type')
        # validation
        result_dict = test(args, model, x_valid, y_valid,
                           'Train epoch ' + str(epoch) +':\t', args.show_process,
                           epoch_start_time, device,result_path=result_path,y_train_mean=y_mean,y_train_std=y_std)

        # print("loss_mse:{}, loss_std:{}, loss_domain:{}".format(loss_mse.item(), loss_std.item(), loss_ba.item()))

        # if args.is_ood:
        #     cal_worst_acc(args,data_packet,model,result_dict,epoch_start_time,ts_data,device)
        #     worst_test_loss_log.append(result_dict['worst_rmse']**2)

        if result_dict['mse'] <= best_mse:
            best_mse = result_dict['mse']
            best_mse_model = copy.deepcopy(model)
            print(f'update best mse! epoch = {epoch}')

        if result_dict['r']**2 >= best_r2:
            best_r2 = result_dict['r']**2
            best_r2_model = copy.deepcopy(model)
    if num_epochs==0:
        best_r2_model = copy.deepcopy(model)
        best_mse_model = copy.deepcopy(model)
        load_model(best_r2_model,modeldir,'best_r2_model.pth')
        load_model(best_mse_model, modeldir, 'best_mse_model.pth')
    else:
        save_model(best_r2_model, modeldir, 'best_r2_model.pth')
        save_model(best_mse_model, modeldir, 'best_mse_model.pth')
    if 'lp' in args.name:
        best_mse_model, best_r2_model = train_downstream(args, x_train, y_train, x_valid, y_valid, best_r2_model, ind_same,
                                                     ind_diff, best_mse_model, need_shuffle, iteration,
                                                     mixup_idx_sample_rate, ts_data, loss_fun, steps_per_epoch,
                                                     step_print_num, device,best_mse, best_r2)

    best_model = best_mse_model if args.metrics=='rmse' else best_r2_model
    visualization(args,best_model,data_packet,device,result_path)
    writer.close()
    return best_mse_model, best_r2_model



def train_downstream(args,x_train,y_train,x_valid,y_valid,best_r2_model,ind_same,ind_diff,best_mse_model,need_shuffle,iteration,
             mixup_idx_sample_rate,ts_data,loss_fun,steps_per_epoch,step_print_num,device, best_mse, best_r2):
    best_mse = 1e10  # for best update
    best_r2 = 0.0
    repr_flag = 1
    print('---------------------down stream----------')
    if 'svd' in args.name:
        print('using mixup distribution')
    if args.metrics == 'r':
        model = copy.deepcopy(best_r2_model)
    else:
        model = copy.deepcopy(best_mse_model)
    # _, optim2 = model.opim_design(args, rate=10)
    base_align = BasisAlignment(model.n_feat, args.batch_size,
                                y_train.shape[-1] if len(y_train.shape) > 1 else 1, device)
    # adv = adv_loss(args,loss_fun)
    # adv.to(device)
    rml = RegMetricLoss()
    model.freeze_upper()
    optim2 = Adam([
        {'params':model.parameters(),'lr':args.optimiser_args['lr']*args.lp_lr_rate},
        # {'params': gaussian_approximation.parameters(), 'lr': args.optimiser_args['lr'] * 10}
    ])
    # optim_dis = adv.optimizer
    need_shuffle=True
    # ind_l, ind_g, mask_label = get_label_pairs(args,y_train,device)
    # if not isinstance(ind_l, np.ndarray):
    #     ind_l = ind_l.cpu().numpy()
    #     ind_g = ind_g.cpu().numpy()
    # ind_l, ind_g = np.arange(len(x_train)),np.arange(len(x_train))
    RankNcon = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')
    print("batchsize is {}".format(args.batch_size))
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        shuffle_idx = np.random.permutation(np.arange(len(x_train)))

        if need_shuffle:  # id
            x_train_input = x_train[shuffle_idx]
            y_train_input = y_train[shuffle_idx]
            # ind_l = ind_l[shuffle_idx]
            # ind_g = ind_g[shuffle_idx]
            # mask_label = mask_label[list(shuffle_idx)]
        else:  # ood
            x_train_input = x_train
            y_train_input = y_train
        loss_mse = torch.zeros(1)
        loss_tri = torch.zeros(1)
        for idx in range(iteration):
            lambd = np.random.beta(args.mix_alpha, args.mix_alpha)

            if need_shuffle:  # get batch idx
                idx_1 = shuffle_idx[idx * args.batch_size:(idx + 1) * args.batch_size]
            else:
                idx_1 = np.arange(len(x_train))[idx * args.batch_size:(idx + 1) * args.batch_size]

            if not mixup_idx_sample_rate is None:
                idx_2 = np.array(
                    [np.random.choice(np.arange(x_train.shape[0]), p=mixup_idx_sample_rate[sel_idx]) for sel_idx in
                     idx_1])
            else:  # random mix
                idx_2 = np.array(
                    [np.random.choice(np.arange(x_train.shape[0])) for sel_idx in idx_1])
            x_input_tmp = x_train_input[idx_1]
            y_input_tmp = y_train_input[idx_1]
            x_input2 = x_train_input[idx_2]
            y_input2 = y_train_input[idx_2]
            # mask_l = mask_label[idx_1]
            # to device
            ####the original version on RCF-MNIST did not use torch.tensor here

            X1 = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
            Y1 = torch.tensor(y_input_tmp, dtype=torch.float32).to(device)
            # Xl = torch.tensor(x_l, dtype=torch.float32).to(device)
            # Yl = torch.tensor(y_l, dtype=torch.float32).to(device)
            # Xg = torch.tensor(x_g, dtype=torch.float32).to(device)
            # Yg = torch.tensor(y_g, dtype=torch.float32).to(device)
            X2 = torch.tensor(x_input2, dtype=torch.float32).to(device)
            Y2 = torch.tensor(y_input2, dtype=torch.float32).to(device)
            # #  mixup
            mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
            mixup_X = X1 * lambd + X2 * (1 - lambd)
            if 'rankncon' in args.name:
                X1_ = X1
                X1 = torch.cat([X1,X1_],dim=0)
                X2_ = X2
                X2 = torch.cat([X2, X2_], dim=0)
                mixup_X_ = mixup_X
                mixup_X = torch.cat([mixup_X,mixup_X_],dim=0)
            if args.use_manifold == True:
                pred_mixup_Y, feats_mixup = model.forward_mixup(X1, X2, lambd, return_feats=True)
            else:
                pred_mixup_Y, feats_mixup = model.forward(mixup_X, return_feats=True)

            pred_Y, feats = model.forward(X1, return_feats=True)
            # pred_l, feats_l = model.forward(Xl, return_feats=True)
            # pred_g, feats_g = model.forward(Xg, return_feats=True)
            if 'rankncon' in args.name:
                feats1, feats2 = torch.split(feats, [args.batch_size, args.batch_size], dim=0)
                feats_mixup1, feats_mixup2 = torch.split(feats_mixup, [args.batch_size, args.batch_size], dim=0)
                feats_new = torch.cat([feats1.unsqueeze(1), feats2.unsqueeze(1)], dim=1)
                feats_mixup_new = torch.cat([feats_mixup1.unsqueeze(1), feats_mixup2.unsqueeze(1)], dim=1)
                pred_Y, _ = torch.split(pred_Y, [args.batch_size, args.batch_size], dim=0)
                pred_mixup_Y, _ = torch.split(pred_mixup_Y, [args.batch_size, args.batch_size], dim=0)
                feats = feats1
                feats_mixup = feats_mixup1
            if args.dataset == 'TimeSeires':  # time series loss need scale
                scale = ts_data.scale.expand(pred_mixup_Y.size(0), ts_data.m)
                loss_mse = loss_fun(pred_mixup_Y * scale, mixup_Y * scale)
            else:
                loss_mse = loss_fun(pred_mixup_Y, mixup_Y)
            loss_mse += loss_fun(pred_Y, Y1)

            std1 = base_align.std_svd2(feats,Y1)
            std2 = base_align.std_svd2(feats_mixup, mixup_Y)
            _,svd1,_ = torch.linalg.svd(feats)
            _,svd2,_ = torch.linalg.svd(feats_mixup)
            loss_std = std1 + std2
            loss_svd = torch.abs(svd1[0] - svd2[0]).mean()
            # loss_rml = rml(feats, Y1)[0]+rml(feats_mixup,mixup_Y)[0]

            loss = loss_mse
            if 'svd' in args.name:
                loss+=args.svd_weight * loss_svd
            if 'std' in args.name:
                loss+= args.std_weight* loss_std
            if 'ranksim' in args.name:
                loss_ranksim = batchwise_ranking_regularizer(feats, Y1, 2) + batchwise_ranking_regularizer(feats_mixup,                                                                                                mixup_Y, 2)
                loss+=loss_ranksim
            if 'rml' in args.name:
                loss_rml = rml(feats, Y1)[0] + rml(feats_mixup, mixup_Y)[0]
                loss+= loss_rml
            if 'rankncon' in args.name:
                loss_rncon = RankNcon(feats_new, Y1) + RankNcon(feats_mixup_new, mixup_Y)
                loss+=loss_rncon



            # loss_tri = tri_loss(feat, feat_s, feat_d, y_input_tmp, y_same_domain_tmp, y_diff_domain_tmp)
            optim2.zero_grad()
            loss.backward()
            optim2.step()

            if args.dataset == 'Dti_dg' and (idx - 1) % (iteration // step_print_num) == 0:
                result_dict = test(args, model, x_valid, y_valid,
                                   'Train epoch ' + str(epoch) + ', step = {} '.format(
                                       (epoch * steps_per_epoch + idx)) + ':\t', args.show_process,
                                   epoch_start_time, device)

                # save best model
                if result_dict['mse'] <= best_mse:
                    best_mse = result_dict['mse']
                    best_mse_model = copy.deepcopy(model)
                if result_dict['r'] ** 2 >= best_r2:
                    best_r2 = result_dict['r'] ** 2
                    best_r2_model = copy.deepcopy(model)
        print("loss_mse:{}, loss_std:{}, loss_con:{}".format(loss_mse.item(), loss_std.item(), loss_svd.item()))
        # print("loss_mse:{}, loss_ranksim:{}".format(loss_mse.item(), loss_ranksim.item()))
        result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) + ':\t', args.show_process,
                           epoch_start_time, device)

        # if args.is_ood:
        #     cal_worst_acc(args,data_packet,model,result_dict,epoch_start_time,ts_data,device)
        #     worst_test_loss_log.append(result_dict['worst_rmse']**2)

        if result_dict['mse'] <= best_mse:
            best_mse = result_dict['mse']
            best_mse_model = copy.deepcopy(model)
            print(f'update best mse! epoch = {epoch}')

        if result_dict['r'] ** 2 >= best_r2:
            best_r2 = result_dict['r'] ** 2
            best_r2_model = copy.deepcopy(model)

    return best_mse_model, best_r2_model