import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn import manifold
from colors import colors

def draw_tsne(features,labels, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = os.path.join(path,name+"_tsne.pdf")
    x = features.cpu().detach().numpy()
    y = labels.cpu().long().squeeze().numpy()
    '''t-sne'''
    tsne = manifold.TSNE(n_components=2,init='pca',random_state=501)
    X_tsne = tsne.fit_transform(x)
    print("original dimention is {}. Embedding dimention is {}".format(x.shape[-1],X_tsne.shape[-1]))

    '''Visualization'''
    x_min,x_max = X_tsne.min(0),X_tsne.max(0)
    x_norm = (X_tsne-x_min)/(x_max-x_min)
    plt.figure(figsize=(16,16))
    # colors = [plt.cm.Set1(i) for i in range(len(np.unique(y)))]
    for i in range(x_norm.shape[0]):
        plt.text(x_norm[i,0],x_norm[i,1],'*',color=plt.cm.Set1(y[i]))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(savepath)
    plt.close()


def visualization(args, model, data_packet, device, path):
    x_train = data_packet['x_train']
    y_train = data_packet['y_train']
    x_test = data_packet['x_test']
    y_test = data_packet['y_test']
    if 'domain_y_train' in data_packet.keys():
        domain_y_train = data_packet['domain_y_train']
    iteration = len(x_train) // args.batch_size
    train_feats = []
    train_labels=[]
    train_domainlabels=[]
    test_feats = []
    model.eval()
    with torch.no_grad():
        for idx in range(iteration):
            idxs = np.arange(len(x_train))[idx * args.batch_size:(idx + 1) * args.batch_size]
            x_train_tmp = x_train[idxs]
            y_train_tmp = y_train[idxs]

            x_train_tmp = torch.tensor(x_train_tmp,dtype=torch.float32).to(device)
            y_train_tmp = torch.tensor(y_train_tmp,dtype=torch.float32)

            _, feat_tmp = model(x_train_tmp,return_feats=True)
            train_feats.append(feat_tmp.cpu())
            train_labels.append(y_train_tmp)
            if 'domain_y_train' in data_packet.keys():
                domain_y_train_tmp = domain_y_train[idxs]
                domain_y_train_tmp = torch.tensor(domain_y_train_tmp,dtype=torch.float32)
                train_domainlabels.append(domain_y_train_tmp)
        train_feats = torch.cat(train_feats,dim=0)
        train_flag = torch.zeros(train_feats.size(0))
        # train_labels = torch.cat(train_labels,dim=0)
        if 'domain_y_train' in data_packet.keys():
            train_domainlabels = torch.cat(train_domainlabels, dim=0)
            draw_tsne(train_feats,train_domainlabels,path,'train_domain')
        if args.dataset == 'Dti_dg':
            val_iter = x_test.shape[0] // args.batch_size
            val_len = args.batch_size
            y_test = y_test[:val_iter * val_len]
        else: # read in the whole test data
            val_iter = 1
            val_len = x_test.shape[0]
        for idx in range(val_iter):
            if isinstance(x_test, np.ndarray):
                x_list_torch = torch.tensor(x_test[idx * val_len:(idx + 1) * val_len], dtype=torch.float32).to(device)
            else:
                x_list_torch = x_test[idx * val_len:(idx + 1) * val_len].to(device)
            model = model.to(device)
            _, feats_tmp = model(x_list_torch, return_feats=True)
            # feats = feats.cpu().numpy()
            test_feats.append(feats_tmp.cpu())
        test_feats = torch.cat(test_feats, dim=0)
        test_flag = torch.ones(test_feats.size(0))
        all_feats = torch.cat([train_feats, test_feats], dim=0)
        all_flags = torch.cat([train_flag, test_flag], dim=0)
        draw_tsne(all_feats,all_flags,path,'train_test')



def save_model(model, modeldir, name):
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    torch.save(model.state_dict(), os.path.join(modeldir, name))

def load_model(model, modeldir, name):
    model.load_state_dict(torch.load(os.path.join(modeldir, name)))



