import torch
from torch import nn
import torch.nn.functional as F

import math

from models import Discriminator

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

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
def pairwise_squared_euclidean_dist(x,y):
    return torch.square(x-y).sum(dim=1)
def s_gradient(dist_map, label_map):
    '''
    all input should be Nx(N-1), N is batch size
    '''
    N = dist_map.size(0)
    mask = label_map.le(1e-12)
    label_map = label_map.clone()
    label_map[mask] = 1e12
    close_x_dist, ind_x = torch.min(label_map, dim=1)
    close_y_dist, ind_y = torch.min(label_map, dim=0)
    s_close_x = dist_map[torch.arange(N).to(label_map).long(), ind_x].unsqueeze(1).expand(N,N-1).detach()
    s_close_y = dist_map[ind_y, torch.arange(N-1).to(label_map).long()].unsqueeze(0).expand(N,N-1).detach()
    gradient_x = (dist_map - s_close_x)/close_x_dist.unsqueeze(1).expand(N,N-1)
    gradient_y = (dist_map - s_close_y)/close_y_dist.unsqueeze(0).expand(N,N-1)
    return gradient_x+gradient_y
def f_continuous(dist_map_, labels):
    N = labels.size(0)
    if len(dist_map_.size())<=1:
        dist_map = dist_map_.view(N,N-1)
    else:
        dist_map = dist_map_
    if len(labels.size()) == 1:
        labels = labels.view(N, 1)
    label_map = l1_dist(labels, labels)
    label_map = label_map.flatten()[:-1].view(N - 1, N + 1)[:, 1:].flatten().view(N, N - 1)
    mask_l = label_map.le(1e-12)
    mask_g = label_map.ge(-1e-12)
    label_map_l = label_map.clone()
    label_map_g = label_map.clone()
    label_map_l[mask_l] = 1e12
    label_map_g[mask_g] = -1e12
    l_dist, ind_x_l = torch.min(label_map_l, dim=1)
    g_dist, ind_x_g = torch.max(label_map_g, dim=1)
    mask_l_ = l_dist.eq(1e12)
    mask_g_ = g_dist.eq(-1e12)
    mask = mask_g_ | mask_l_
    gradient_l = dist_map[torch.arange(N).to(label_map).long(), ind_x_l]
    gradient_g = dist_map[torch.arange(N).to(label_map).long(), ind_x_g]
    loss = torch.abs(gradient_l[~mask]-gradient_g[~mask])

    return loss.mean()
def f_continuous2(dist_map, labels1, labels2):
    N = labels1.size(0)
    if len(labels1.size()) == 1:
        labels1 = labels1.view(N, 1)
        labels2 = labels2.view(N, 1)
    label_map = l1_dist(labels1, labels2)
    mask_l = label_map.le(1e-12)
    mask_g = label_map.ge(-1e-12)
    label_map_l = label_map.clone()
    label_map_g = label_map.clone()
    label_map_l[mask_l] = 1e12
    label_map_g[mask_g] = -1e12
    l_dist, ind_x_l = torch.min(label_map_l, dim=1)
    g_dist, ind_x_g = torch.max(label_map_g, dim=1)
    mask_l_ = l_dist.eq(1e12)
    mask_g_ = g_dist.eq(-1e12)
    mask = mask_g_ | mask_l_
    weights = g_dist+l_dist
    l_weight = g_dist/ (g_dist+l_dist)
    g_weight = l_dist/ (g_dist+l_dist)
    weights = normalize(weights, axis=0)
    gradient_l = dist_map[torch.arange(N).to(label_map).long(), ind_x_l] *l_weight
    gradient_g = dist_map[torch.arange(N).to(label_map).long(), ind_x_g] *g_weight
    loss = gradient_l[~mask]-gradient_g[~mask]
    weights = weights[~mask]
    loss = torch.abs(weights*loss)
    # loss_mask = loss.ge(1e9) | loss.le(-1e9)
    # loss = loss[~loss_mask]
    return loss.mean()


def s_continuous_grad(dist_map, labels):
    '''
    labels should be N, others should be Nx(N-1)
    '''
    N = dist_map.size(0)
    if len(labels.size())==1:
        labels = labels.view(N,1)
    label_map = l1_dist(labels,labels)
    label_map = label_map.flatten()[:-1].view(N - 1, N + 1)[:, 1:].flatten().view(N, N - 1)
    mask_l = label_map.le(1e-12)
    mask_g = label_map.ge(-1e-12)
    label_map_l = label_map.clone()
    label_map_g = label_map.clone()
    label_map_l[mask_l] = 1e12
    label_map_g[mask_g] = -1e12
    close_x_l, ind_x_l = torch.min(label_map_l,dim=1)
    close_y_l, ind_y_l = torch.min(label_map_l,dim=0)
    close_x_g, ind_x_g = torch.max(label_map_g,dim=1)
    close_y_g, ind_y_g = torch.max(label_map_g,dim=0)
    s_close_x_l = dist_map[torch.arange(N).to(label_map).long(), ind_x_l].unsqueeze(1).expand(N, N - 1).detach()
    s_close_y_l = dist_map[ind_y_l, torch.arange(N - 1).to(label_map).long()].unsqueeze(0).expand(N, N - 1).detach()
    s_close_x_g = dist_map[torch.arange(N).to(label_map).long(), ind_x_g].unsqueeze(1).expand(N, N - 1).detach()
    s_close_y_g = dist_map[ind_y_g, torch.arange(N - 1).to(label_map).long()].unsqueeze(0).expand(N, N - 1).detach()
    gradient_x_l = (dist_map - s_close_x_l) / close_x_l.unsqueeze(1).expand(N, N - 1)
    gradient_y_l = (dist_map - s_close_y_l) / close_y_l.unsqueeze(0).expand(N, N - 1)
    gradient_x_g = (s_close_x_g - dist_map) / close_x_g.unsqueeze(1).expand(N, N - 1)
    gradient_y_g = (s_close_y_g - dist_map) / close_y_g.unsqueeze(0).expand(N, N - 1)
    loss_x = gradient_x_l+gradient_x_g
    loss_y = gradient_y_l+gradient_y_g
    return loss_x.clamp(min=0).mean()+loss_y.clamp(min=0).mean()
def s_continuous(feats, feats_mixup, labels, labels_mixup):
    '''
    label -> Bx1
    feature ->BXN
    '''
    dist_map = euclidean_dist(feats, feats).view(-1,1)
    mixup_dist_map = euclidean_dist(feats_mixup,feats_mixup).view(-1,1)
    label_map = euclidean_dist(labels,labels)
    mixup_label_map = euclidean_dist(labels_mixup,labels_mixup)
    labe_dd_map = euclidean_dist(label_map.view(-1,1), mixup_label_map.view(-1,1))
    min_d, ind_dist = torch.min(labe_dd_map, dim=1)
    mix_min_d, mixup_ind_dist = torch.min(labe_dd_map, dim=0)
    loss = torch.abs(dist_map-mixup_dist_map[ind_dist])\
           +torch.abs(mixup_dist_map - dist_map[mixup_ind_dist])

    return loss.mean()



def square_div(x,y):
    """
    Args:
      x: pytorch Variable, with shape [n, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [n, 1]
    """
    x_y = x-y
    dist = torch.sum(torch.mul(x_y,x_y),dim=1)
    dist = dist.clamp(min=1e-12)  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

def IS_divergence(x,y):
    div = y/x
    dis = torch.sum(div) - torch.prod(y.size())-torch.sum(torch.log(div))


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    #
    # # `dist_ap` means distance(anchor, positive)
    # # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # dist_ap, relative_p_inds = torch.max(
    #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    # dist_an, relative_n_inds = torch.min(
        # dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_aps,dist_ans = [],[]
    for i in range(N):
        dist_ap= torch.mean(dist_mat[i][is_pos[i]].contiguous(), 0, keepdim=True)
        dist_an = torch.mean(dist_mat[i][is_neg[i]].contiguous(), 0, keepdim=True)
        dist_aps.append(dist_ap)
        dist_ans.append(dist_an)
    # shape [N]
    dist_aps = torch.cat(dist_aps).clamp(min=1e-12, max=1e12)
    dist_ans = torch.cat(dist_ans).clamp(min=1e-12)

    return dist_aps, dist_ans

def domain_dispersion(fix_dist_map,dist_map,label):
    N = fix_dist_map.size(0)
    eye = torch.eye(N).to(fix_dist_map)
    fix_dist_map_ = fix_dist_map+eye*1e12
    _, is_same = torch.min(fix_dist_map_,dim=1)
    _, is_diff = torch.max(fix_dist_map,dim=1)
    # b_map = dist_map[is_same]
    # b_label = label[is_same]
    ap_dist = dist_map.gather(dim=1,index=is_diff.view(-1,1))
    an_dist = dist_map.gather(dim=1,index=is_same.view(-1,1))
    return ap_dist, an_dist

def squared_Hellinger_distance(x,y):
    """
    Args:
      x: pytorch Variable, with shape [n, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [n, 1]
    """
    # x = F.relu(x)
    # y = F.relu(y)
    x = torch.sqrt(x)
    y = torch.sqrt(y)
    out = torch.sum(torch.square(x-y),dim=1,keepdim=True)
    # out = F.pairwise_distance(x,y,p=2)
    return out
def pair_mining(feats, labels):
    N = feats.size(0)
    feat_dist_map = euclidean_dist(feats, feats)
    if len(labels.size()) < 2:
        labels = labels.unsqueeze(1)
    label_dist_map = euclidean_dist(labels, labels)
    dist_map = feat_dist_map / (label_dist_map + 1e-12)  # size NxN
    eye = torch.eye(N).to(dist_map)
    dist_map_1 = dist_map + (eye * 1e12)
    dist_map_2 = dist_map * (1 - eye)
    _, ind_same = torch.min(dist_map_1, dim=1)
    _, ind_diff = torch.max(dist_map_2, dim=1)

class DomainTripletLoss(object):
    def __init__(self,margin=0, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, s_features,d_features, y,ys,yd):
        global_feat = normalize(global_feat, axis=-1)
        s_features = normalize(s_features,axis=-1)
        d_features = normalize(d_features,axis=-1)
        if len(y.size())<2:
            y = y.unsqueeze(1)
            ys = ys.unsqueeze(1)
            yd = yd.unsqueeze(1)
        dist_ap = F.pairwise_distance(global_feat,d_features,p=2)
        # l_dist_ap = torch.abs(y-yd).squeeze()
        # l_dist_ap = F.pairwise_distance(y,yd,p=2)/
        # dist_ap = dist_ap/(l_dist_ap+1e-12)
        dist_an = F.pairwise_distance(global_feat,s_features,p=2)
        # l_dist_an = torch.abs(y-ys).squeeze()
        l_dist_an = F.pairwise_distance(y,ys,p=2)
        # print(l_dist_an)
        dist_an = dist_an / (l_dist_an+1e-12)
        loss = self.margin/(dist_an+1e-12)
        # fakev = torch.ones(dist_ap.size()).to(dist_ap)
        # dists = torch.stack([-,-dist_an],dim=1)
        # labels = torch.zeros(dists.size(0)).to(dists).long()
        # loss = F.cross_entropy(torch.softmax(dists,dim=1),labels)

        return loss.mean()

    # def __call__(self, global_feat, s_features,d_features, y,ys,yd):
    #     global_feat = normalize(global_feat, axis=-1)
    #     s_features = normalize(s_features,axis=-1)
    #     d_features = normalize(d_features,axis=-1)
    #     dist_ap = F.pairwise_distance(global_feat,d_features,p=2)
    #     l_dist_ap = torch.abs(y-yd).squeeze()
    #     dist_ap = dist_ap/(l_dist_ap+1e-12)
    #     dist_an = F.pairwise_distance(global_feat,s_features,p=2)
    #     l_dist_an = torch.abs(y-ys).squeeze()
    #     dist_an = dist_an / (l_dist_an+1e-12)
    #
    #     dists = torch.stack([dist_ap,dist_an],dim=1)
    #     labels = torch.ones(dists.size(0)).to(dists).long()
    #     loss =  F.cross_entropy(torch.softmax(dists,dim=1),labels)
    #
    #     return loss
class ContrastiveRegression(object):
    def __init__(self, tau):
        self.tau = tau
    def __call__(self, x, y):
        N = x.size(0)
        label_dist = euclidean_dist(y.view(-1,1),y.view(-1,1))
        label_dist_sorted,sort_ind= torch.sort(label_dist,dim=1,descending=True)
        r_ind = torch.arange(N).view(-1,1).expand(N,N)
        feat_dist = euclidean_dist(x,x)
        feat_dist = torch.exp(-feat_dist/self.tau)
        soreted_feat_dist = feat_dist[r_ind,sort_ind]
        csum_feat_dist = torch.cumsum(soreted_feat_dist,dim=1)
        soreted_feat_dist = soreted_feat_dist[:,1:]
        csum_feat_dist = csum_feat_dist[:,:-1]
        return -torch.log(soreted_feat_dist/csum_feat_dist).mean(dim=1).mean()

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
class GaussianApproximation(nn.Module):
    def __init__(self,featsize,label_size,device,args):
        super(GaussianApproximation, self).__init__()
        self.feat_size = featsize
        self.device = device
        self.label_size= label_size
        self.alpha = 1 / 2*math.pi
        # self.register_parameter('alpha', nn.Parameter((torch.ones(1) / (1 / 2*math.pi)).to(device)))
        # self.register_parameter('eps', nn.Parameter(torch.eye(label_size+1).to(device)))
        # self.register_parameter('mu', nn.Parameter(torch.zeros(label_size+1,1).to(device)))
        self.args = args
        if args.need_bias:
            self.register_parameter('bias', nn.Parameter(0.6*torch.ones(featsize).to(device)))
        self.mse = nn.MSELoss()
    def init_gaussian_distribution(self,labels,edge=1.0):
        datanum = labels.size(0)
        unknown_variable = torch.linspace(start=-edge,end=edge,steps=self.feat_size).\
            view(1,self.feat_size).expand(datanum,self.feat_size).unsqueeze(1)
        unknown_variable = unknown_variable.to(self.device)
        known_variable = labels.view(-1,self.label_size).unsqueeze(-1).expand(-1, self.label_size, self.feat_size)
        variable = torch.cat([known_variable,unknown_variable], dim=1) #bx(labelsize+1)xfeatsize
        # variable = variable - self.mu
        # variable = torch.einsum('jii->ji',variable.transpose(2,1)@self.eps@variable)
        # return torch.exp(-0.5*variable)
        return torch.exp(-0.5*torch.sum(variable*variable, dim=1))
    def forward(self, batch_feat, batch_y,edge=1.0):
        gaussian_feat = self.init_gaussian_distribution(batch_y,edge=edge)
        # print("y:{}".format(batch_y))
        # print("gaussian:{}".format(gaussian_feat[1]))
        # loss = F.pairwise_distance(batch_feat, self.alpha*gaussian_feat, p=2).mean()
        if self.args.need_bias:
            batch_feat = batch_feat+self.bias
        # print("batch_feats:{}".format(batch_feat))
        # print("gaussian{}".format(gaussian_feat))
        # print(self.alpha)
        loss = self.mse(self.alpha*batch_feat,gaussian_feat)#+ self.mmd(batch_feat, gaussian_feat)
        if self.args.need_alpha:
            self.alpha = (torch.mean(gaussian_feat) / (torch.mean(batch_feat)+1e-12)).item()
        return loss,gaussian_feat
    @staticmethod
    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                           int(total.size(0)), \
                                           int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                           int(total.size(0)), \
                                           int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

        # bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                      bandwidth_temp in bandwidth_list]

        return sum(kernel_val)
    def mmd(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss,Source<->Source
        XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st，Source<->Target

        YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts,Target<->Source
        YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt,Target<->Target

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss

class BasisAlignment(torch.nn.Module):
    def __init__(self,featsize,batchsize,label_size,device):
        super(BasisAlignment, self).__init__()
        # self.proto_basis = nn.Parameter(torch.ones(featsize,batchsize).to(device))
        self.alpha = nn.Parameter(torch.ones(1).to(device))
        self.batch_size = batchsize
        self.feat_size = featsize
        self.label_size = label_size
        self.device = device
        self.s_value = 1.0
    def init_gaussian_distribution(self,labels,edge=1.0):
        datanum = labels.size(0)
        unknown_variable = torch.linspace(start=-edge,end=edge,steps=self.feat_size).\
            view(1,self.feat_size).expand(datanum,self.feat_size).unsqueeze(1)
        unknown_variable = unknown_variable.to(self.device)
        known_variable = labels.view(-1,self.label_size).unsqueeze(-1).expand(-1, self.label_size, self.feat_size)
        variable = torch.cat([known_variable,unknown_variable], dim=1) #bx(labelsize+1)xfeatsize
        # variable = variable - self.mu
        # variable = torch.einsum('jii->ji',variable.transpose(2,1)@self.eps@variable)
        # return torch.exp(-0.5*variable)
        return torch.exp(-0.5*torch.sum(variable*variable, dim=1))
    def calculate_basis(self,feats_a, feats_n, label_a):
        feats_a = self.standardization(feats_a)
        feats_n = self.standardization(feats_n)
        u_a,s_a,v_a = torch.svd(feats_a.t())
        u_n,s_n,v_n = torch.svd(feats_n.t())
        # gaussian_feat = self.init_gaussian_distribution(label_a)
        # print(gaussian_feat)
        # u_n,s_n,v_n = torch.svd(gaussian_feat.t())
        # print(feats_a * u_n.t())
        c1 = (feats_a * u_n.t()).sum(dim=-1)*(1 / self.batch_size)
        loss1 = c1.add_(-1.0).pow_(2).sum()
        c2 = (feats_n * u_a.t()).sum(dim=-1) * (1 / self.batch_size)
        loss2 = c2.add_(-1.0).pow_(2).sum()
        return loss1+loss2
        # print(u_a.t()@u_a)
        # p_s, cospa, p_t = torch.svd(torch.mm(u_a.t(),u_n))
        # cospa = cospa.clamp(max=1.0)
        # print(cospa)
        # dist_sin = torch.sqrt(1-torch.pow(cospa, 2))
        # return torch.norm(dist_sin, 1)\

    def NOB_loss(self,feats, feats_n, label):
        feats = feats.transpose(0,1)
        feats_n = feats_n.transpose(0,1)
        feats = self.standardization(feats)
        feats_n = self.standardization(feats_n)
        # print(feats)
        # print(feats_n)

        base = self.find_nobs(feats).detach()
        base_n = self.find_nobs(feats_n).detach()

        c1 = (feats * base_n).sum(dim=-1)*(1/self.batch_size)
        loss1 = c1.add_(-1.0).pow_(2).sum()
        # print("loss1{}".format(loss1))
        c2 = (feats_n * base).sum(dim=-1) * (1 / self.batch_size)
        loss2 = c2.add_(-1.0).pow_(2).sum()
        # print("loss2{}".format(loss2))
        return loss1 + loss2
    def align_feat_bases(self,feats1, feats2, labels1, labels2):
        feat_dist_map = l1_dist(feats1, feats2)
        # if len(labels1.size()) == 1:
        #     labels1 = labels1.view(-1, 1)
        #     labels2 = labels2.view(-1, 1)
        # label_dist_map = l1_dist(labels1, labels2)
        # _, s, _ = torch.linalg.svd(dist_map)
        # _,s1,_ = torch.linalg.svd(feats1)
        # _,s2,_ = torch.linalg.svd(feats2)
        # sigma = torch.pow(s1[0],2)+torch.pow(s2[0],2)
        loss_continuous = f_continuous2(feat_dist_map,feats1, feats2)
        return loss_continuous, loss_continuous
    def local_continuous(self, feats, labels):
        B = feats.size(0)
        feat_dist_map = l1_dist(feats, feats)
        feat_dist_map = feat_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten()
        if len(labels.size()) == 1:
            labels = labels.view(-1, 1)
        label_dist_map = l1_dist(labels, labels)
        label_dist_map = label_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten()
        label_dist_map = label_dist_map.clamp(min=1e-12)
        mask = label_dist_map.gt(1e-12)
        dist_map = feat_dist_map / label_dist_map
        loss_local = f_continuous(dist_map, labels)
    def global_continuous(self, feats, featsl, featsg, label, labell, labelg):
        feat_dist_l = feats - featsl
        feat_dist_g = featsg - feats
        label_dist_l = label - labell
        label_dist_g = labelg - label
        feat_dist_l = feat_dist_l.mean(dim=-1)
        feat_dist_g = feat_dist_g.mean(dim=-1)
        weights = label_dist_l + label_dist_g
        l_weight = label_dist_g / weights
        g_weight = label_dist_l / weights
        w = normalize(weights, axis=-1)
        gradient_l = feat_dist_l * l_weight
        gradient_g = feat_dist_g * g_weight
        loss_con = torch.abs(w*(gradient_l-gradient_g))
        return loss_con.mean()
    def svd_std(self,feats, labels):
        B = feats.size(0)
        feat_dist_map = euclidean_dist(feats, feats)
        feat_dist_map = feat_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten()
        if len(labels.size())==1:
            labels = labels.view(-1,1)
        label_dist_map = l1_dist(labels,labels)
        label_dist_map = label_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten()
        label_dist_map = torch.abs(label_dist_map).clamp(min=1e-12)
        mask = label_dist_map.gt(1e-12)
        dist_map = feat_dist_map / label_dist_map
        ret_dist_map = dist_map.view(B,B-1)
        dist_map = dist_map[mask]
        loss_std = torch.std(dist_map)
        _,s,_ = torch.linalg.svd(feats)
        # loss_svd = torch.pow(s[0], 2)
        return loss_std, s, ret_dist_map
    def std_svd2(self,feats,labels):
        B = feats.size(0)
        if len(labels.size())==1:
            labels = labels.view(-1,1)
        ln = labels.size(1)
        label_dist = labels.unsqueeze(1)- labels.unsqueeze(0)
        label_dist_map = label_dist.view(B*B,ln)[:-1,:].view(B - 1, B + 1,ln)[:, 1:]
        label_dist_map = torch.abs(label_dist_map).clamp(min=1e-12)
        feat_dist_map = euclidean_dist(feats, feats)
        feat_dist_map = feat_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].unsqueeze(2)
        feat_dist_map = feat_dist_map.expand(label_dist_map.size())
        dist_map = feat_dist_map / label_dist_map
        dist_map = dist_map.reshape(B*(B-1), ln)
        t_label_dist = label_dist_map.reshape(B*(B-1),ln)
        std_loss = []
        for i in range(ln):
            mask = t_label_dist[:,i].gt(1e-12)
            std_t = torch.std(dist_map[:,i][mask])
            std_loss.append(std_t)
        std_loss = sum(std_loss)*(1.0/ln)
        return std_loss



    def align_feat_and_label(self,feats,labels):
        B = feats.size(0)
        feat_dist_map = l1_dist(feats,feats)
        feat_dist_map = feat_dist_map.flatten()[:-1].view(B-1,B+1)[:,1:].flatten()
        if len(labels.size())==1:
            labels = labels.view(-1,1)
        label_dist_map = torch.abs(l1_dist(labels,labels))
        # gaussian_feat = self.init_gaussian_distribution(labels)
        # label_dist_map = euclidean_dist(gaussian_feat,gaussian_feat)
        label_dist_map = label_dist_map.flatten()[:-1].view(B-1,B+1)[:,1:].flatten()
        label_dist_map = label_dist_map.clamp(min=1e-12)
        mask = label_dist_map.gt(1e-12)
        dist_map = feat_dist_map / label_dist_map
        loss_local = f_continuous(dist_map, labels)
        # dist_matrix_max = dist_map.abs().clone().view(B,B-1)
        # # dist_matrix_min = dist_map.clone().view(B, B - 1)
        # mask_matrix = label_dist_map.le(1e-12).view(B,B-1)
        # dist_matrix_max[mask_matrix] = -1e12
        # dist_matrix_min[mask_matrix] = 1e12
        # max_s,_ = torch.max(dist_matrix_max,dim=1,keepdim=True)
        # min_s,_ = torch.min(dist_matrix_min,dim=1,keepdim=True)
        dist_map = dist_map[mask]
        # loss = torch.abs(torch.max(dist_map)- torch.min(dist_map))
        # y = max_s.new().resize_as_(max_s).fill_(1)
        # loss_contrast = F.margin_ranking_loss(min_s,max_s,y,margin=0)
        # loss_domain = max_s.abs().mean()
        loss_std = torch.std(dist_map)
        # self.s = torch.mean(dist_map)

        # loss = loss_std+loss_contrast

        return loss_std, loss_local

    def align_continuous_s(self,feats,labels):
        B = feats.size(0)
        feat_dist_map = euclidean_dist(feats, feats)
        feat_dist_map = feat_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten().view(B,B-1)
        loss = s_continuous_grad(feat_dist_map,labels)
        return loss

    def align_feats_and_labels(self,feats,feats_n,labels,labels_n):
        B = feats.size(0)
        feat_dist_map = euclidean_dist(feats, feats)
        feat_dist_map = feat_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten().view(B,B-1)
        feats_n_dist_map = euclidean_dist(feats_n,feats_n)
        feats_n_dist_map = feats_n_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten().view(B,B-1)
        if len(labels.size())==1:
            labels = labels.view(-1,1)
            labels_n = labels_n.view(-1,1)
        # label_dist_map = euclidean_dist(labels,labels)
        # label_n_dist_map = euclidean_dist(labels_n,labels_n)
        # label_dist_map = label_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten().view(B,B-1)
        # label_dist_map = label_dist_map.clamp(min=1e-12)
        # label_n_dist_map = label_n_dist_map.flatten()[:-1].view(B - 1, B + 1)[:, 1:].flatten().view(B,B-1)
        # label_n_dist_map = label_n_dist_map.clamp(min=1e-12)
        # label_dist_map = torch.tensor(label_dist_map,dtype=torch.float32)
        # dist_map = feat_dist_map / label_dist_map
        # dist_map_n = feats_n_dist_map / label_n_dist_map
        # loss = torch.abs(self.alpha*torch.mean(dist_map) - torch.mean(dist_map_n))
        # dist_map_gradient = s_gradient(dist_map,label_dist_map)
        # dist_map_n_gradient = s_gradient(dist_map_n,label_n_dist_map)
        # loss = torch.abs(torch.mean(dist_map_gradient)-torch.mean(dist_map_n_gradient))
        # loss =torch.abs(torch.std(dist_map)- torch.std(dist_map_n))
        # loss = torch.sqrt(loss*loss.clone().detach())
        # u,s,v = torch.svd(dist_map.t())
        # u_n,s_n,v_n = torch.svd(dist_map_n.t())
        # p_a,cospa,p_b = torch.svd(torch.mm(u.t(),u_n))
        # sinpa = torch.sqrt(1-torch.pow(cospa, 2))
        # return loss
    def find_nobs(self, x_old):
        feature_dim, batch_size = x_old.size()
        # try:
        x = x_old.view(1, feature_dim, batch_size)
        f_cov = (torch.bmm(x, x.transpose(1, 2)) / (batch_size - 1)).float()  # N * N
        x_stack = torch.FloatTensor().to(f_cov.device)
        for i in range(f_cov.size(0)):
            f_cov = torch.where(torch.isnan(f_cov), torch.zeros_like(f_cov), f_cov)
            U, S, V = torch.svd(f_cov[i])
            diag = torch.diag(1.0 / torch.sqrt(S + 1e-5))
            rotate_mtx = torch.mm(torch.mm(U, diag), U.transpose(0, 1)).detach()  # N * N
            x_transform = torch.mm(rotate_mtx, x[i])
            x_stack = torch.cat([x_stack, x_transform], dim=0)
        return x_stack.detach()
        # except:
        #     return x_old.detach()
    @staticmethod
    def standardization(data, eps=1e-5):
        # N * d
        mu = torch.mean(data, dim=-1, keepdim=True)
        sigma = torch.std(data, dim=-1, keepdim=True)
        return (data - mu) / (sigma + eps)

    @staticmethod
    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                      bandwidth_temp in bandwidth_list]

        return sum(kernel_val)
    def proportion_continuous(self,feats ,mixup_feats, labels,mixup_labels):
        if len(labels.size())==1:
            labels = labels.view(-1,1)
            mixup_labels = mixup_labels.view(-1,1)
        return s_continuous(feats,mixup_feats,labels,mixup_labels)
    def mmd(self, source, target, kernel_mul=2.0, kernel_num=10, fix_sigma=None):
        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = self.guassian_kernel(source, target,
                                       kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss,Source<->Source
        XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st，Source<->Target

        YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts,Target<->Source
        YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt,Target<->Target

        loss = XX.sum() + (XY + YX).sum() + YY.sum()
        return loss

class adv_loss(nn.Module):
    def __init__(self,args, loss_fun):
        super(adv_loss, self).__init__()
        self.discriminator = Discriminator(args.batch_size-1)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(),**args.optimiser_args)
        self.B = args.batch_size
        self.loss_fun = loss_fun
    def forward(self,feat_map1, feat_map2):
        score1 = self.discriminator(feat_map1)
        score2 = self.discriminator(feat_map2)
        loss1 = self.loss_fun(score1, torch.ones(score1.size()).to(score1))
        loss2 = self.loss_fun(score2, torch.zeros(score1.size()).to(score1))

        return (loss1+loss2).mean()


class RegMetricLoss(nn.Module):
    def __init__(self, sigma=1, w_leak=0.2, p=0.9):
        super(RegMetricLoss, self).__init__()
        self.sigma = sigma
        #         self.c = (1/(sigma * np.sqrt(2*np.pi))).tolist()
        self.w_leak = w_leak
        self.alpha = None
        self.p = p
        print('RMLoss: sigma=%f, p=%f' % (self.sigma, self.p))

    def forward(self, feature, label):
        f2 = (feature ** 2).sum(1)
        f_dist = f2.unsqueeze(0) + f2.unsqueeze(
            1) - 2 * torch.matmul(feature, feature.transpose(0, 1))
        f_dist = torch.sqrt(F.relu(f_dist))
        # label_0 = label.unsqueeze(0)
        # label_1 = label.unsqueeze(1)
        if len(label.size())==1:
            label = label.view(-1,1)
        l_dist = euclidean_dist(label , label)
        w = (torch.exp(-((l_dist / self.sigma) ** 2) / 2) + self.w_leak).detach()
        diff = torch.abs(f_dist - l_dist)
        w_diff = w * diff

        mean_diff = w_diff.mean().data.cpu().numpy().tolist()
        if self.alpha is None:
            self.alpha = mean_diff
        else:
            self.alpha = self.p * self.alpha + (1 - self.p) * mean_diff
        mask = w_diff.gt(self.alpha)
        nonz_num = mask.sum().float()
        #         nonz_diff = torch.masked_select(diff, mask)
        corr_w = torch.masked_select(w, mask)
        #         loss = corr_w*nonz_diff
        loss = torch.masked_select(w_diff, mask)
        #         print(self.alpha, mean_diff, nonz_num, corr_w.sum())
        return loss.sum() / (corr_w.sum() + 1e-9), nonz_num, self.alpha, mean_diff
