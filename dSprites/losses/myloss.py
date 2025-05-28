import torch
from torch import nn
import torch.nn.functional as F

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


def std_svd2(feats,labels):
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







