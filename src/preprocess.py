import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn import manifold

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
def l1_dist(x,y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    return (x.unsqueeze(1)-y.unsqueeze(0)).mean(dim=2)
def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def H_dist(x,y):
    m, n = x.size(0), y.size(0)
    x, y = torch.sqrt(x), torch.sqrt(y)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def getDomainPairs(dataset,device,model,path):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        print("extracting features")
        for i,(x,y) in enumerate(dataset):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            _, feat = model(x.unsqueeze(0),return_feats=True)
            feats.append(feat)
            labels.append(y.view(1,-1))
        del x
        del y
        print("preprocessing features")
        feats = torch.cat(feats).cpu()
        labels = torch.cat(labels).cpu()
        N = feats.size(0)
        feat_dist_map = euclidean_dist(feats,feats)
        if len(labels.size())<2:
            labels = labels.unsqueeze(1)
        label_dist_map = euclidean_dist(labels,labels)
        dist_map = feat_dist_map / (label_dist_map+1e-12) # size NxN
        eye = torch.eye(N).to(dist_map)
        dist_map_1 = dist_map+(eye*1e12)
        dist_map_2 = dist_map*(1-eye)
        _,ind_same = torch.min(dist_map_1,dim=1)
        _,ind_diff = torch.max(dist_map_2,dim=1)
        # label_dist_map_ = label_dist_map+eye*1e12
        # _,ind_same = torch.max(label_dist_map,dim=1)
        # _,ind_diff = torch.min(label_dist_map_,dim=1)
        for i in range(N):
            if ind_same[i]==i or ind_diff[i]==i:
                raise ValueError
        if path is not None:
            as_labels = labels[ind_same]
            ad_labels = labels[ind_diff]
            labels = labels.squeeze().numpy()
            as_labels = as_labels.squeeze().numpy()
            ad_labels = ad_labels.squeeze().numpy()
            ad_diff = np.abs(labels - ad_labels)
            as_diff = np.abs(labels - as_labels)
            x = np.linspace(0,N,num=N)
            plt.figure(figsize=(32,32))
            # plt.plot(x,labels,color='red',linewidth=1, label='anchor')
            plt.plot(x,as_diff,color='green',linewidth=1, label='diffclass')
            plt.plot(x,ad_diff,color='blue', linewidth=1, label='sameclass')
            plt.savefig(os.path.join(path,'labeldiff.png'), bbox_inches='tight')
        print("preprocessing done")
    return ind_same ,ind_diff
def get_domain_pairs(args,x_train, y_train, device, model,path=None):
    model.eval()
    feats = []
    labels = []
    datanum = len(x_train)
    iteration = datanum // args.batch_size
    with torch.no_grad():
        print("extracting features")
        for idx in range(iteration):
            x = x_train[idx * args.batch_size:(idx + 1) * args.batch_size]
            y = y_train[idx * args.batch_size:(idx + 1) * args.batch_size]
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            _, feat = model(x, return_feats=True)
            feats.append(feat)
            labels.append(y)
        x = x_train[(idx + 1) * args.batch_size:]
        y = y_train[(idx + 1) * args.batch_size:]
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        _, feat = model(x, return_feats=True)
        feats.append(feat)
        labels.append(y)
        print("preprocessing features")
        feats = torch.cat(feats).cpu()
        labels = torch.cat(labels).cpu()
        N = feats.size()
        if len(labels.size())<2:
            labels = labels.unsqueeze(1)
        feat_dist_map = euclidean_dist(feats, feats)
        feat_dist_map = feat_dist_map.flatten()[:-1].view(datanum - 1, datanum + 1)[:, 1:].flatten().view(datanum, datanum-1)
        label_dist_map= euclidean_dist(labels,labels)
        label_dist_map = label_dist_map.flatten()[:-1].view(datanum-1,datanum+1)[:,1:].flatten().view(datanum,datanum-1)
        label_dist_map = label_dist_map.clamp(min=1e-12)
        dist_map = feat_dist_map / label_dist_map
        _, ind_same = torch.min(dist_map,dim=-1)
        _, ind_diff = torch.max(dist_map,dim=-1)
        if path is not None:
            as_labels = labels[ind_same]
            ad_labels = labels[ind_diff]
            labels = labels.squeeze().numpy()
            as_labels = as_labels.squeeze().numpy()
            ad_labels = ad_labels.squeeze().numpy()
            ad_diff = np.abs(labels - ad_labels)
            as_diff = np.abs(labels - as_labels)
            x = np.linspace(0,datanum,num=datanum)
            plt.figure(figsize=(32,32))
            # plt.plot(x,labels,color='red',linewidth=1, label='anchor')
            plt.plot(x,as_diff,color='green',linewidth=1, label='diffclass')
            plt.plot(x,ad_diff,color='blue', linewidth=1, label='sameclass')
            plt.savefig(os.path.join(path,'labeldiff.png'), bbox_inches='tight')
        print("preprocessing done")
        model.train()
        return ind_same, ind_diff

def get_label_pairs(args, y_train, device, path=None):
    labels = torch.tensor(y_train, dtype=torch.float32)
    label_map = l1_dist(labels, labels)
    mask_l = label_map.le(1e-12)
    mask_g = label_map.ge(-1e-12)
    label_map_l = label_map.clone()
    label_map_g = label_map.clone()
    label_map_l[mask_l] = 1e12
    label_map_g[mask_g] = -1e12
    l_dist, ind_x_l = torch.min(label_map_l, dim=1)
    g_dist, ind_x_g = torch.max(label_map_g, dim=1)
    print(l_dist)
    print(g_dist)
    mask_l_ = l_dist.eq(1e12)
    mask_g_ = g_dist.eq(-1e12)
    mask = mask_g_ | mask_l_
    return ind_x_l, ind_x_g, ~mask

def draw_tsne(features,labels, path):
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = os.path.join(path,"an_tsne.jpg")
    x = features.cpu().detach().numpy()
    y = labels.cpu().numpy()
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
    del x_norm
    del y
    del X_tsne
    del x