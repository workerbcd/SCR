import torch
from torch import nn
import torch.nn.functional as F

import math

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


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

class BasisAlignment(torch.nn.Module):
    def __init__(self,featsize,batchsize,label_size,device):
        super(BasisAlignment, self).__init__()
        self.proto_basis = nn.Parameter(torch.ones(featsize,batchsize).to(device))
        self.batch_size = batchsize
        self.feat_size = featsize
        self.label_size = label_size
        self.device = device
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
        print(feats_a)
        print(feats_n)
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

    def q2i(self, feats, feats_n, label):
        gaussian_feat = self.init_gaussian_distribution(label)
        feat = gaussian_feat @ feats.t()
        try:
            u, s, v = torch.linalg.svd(feat)
            print("s value{}".format(s))
            print("feat {}".format(feat))
            raise ValueError
        except:
            # print(feats)
            # print(feats_n)
            raise ValueError
        c = u @ v.t()
        target = torch.eye(c.size(0)).to(self.device)
        return torch.norm(c-target,2)

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