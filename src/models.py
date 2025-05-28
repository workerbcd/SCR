import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import random 
import data.Dti_dg_lib.networks as networks
from copy import deepcopy

from numbers import Number


class Learner(nn.Module):
    def __init__(self, args, hid_dim = 128, weights = None):
        super(Learner, self).__init__()
        self.block_1 = nn.Sequential(nn.Linear(args.input_dim, hid_dim), nn.LeakyReLU(0.1))
        self.block_2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.LeakyReLU(0.1))
        self.fclayer = nn.Sequential(nn.Linear(hid_dim, 1))
        self.n_feat = hid_dim
        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward_mixup(self, x1, x2, lam=None,return_feats=False,use_FDS=False, targets=None):
        x1 = self.block_1(x1)
        x2 = self.block_1(x2)
        x = lam * x1 + (1 - lam) * x2
        x = self.block_2(x)
        if use_FDS:
            assert targets!=None
            x = self.FDS.smooth(x,targets)
        output = self.fclayer(x)

        if return_feats:
            return output,x
        return output

    def forward(self, x,return_feats=False,use_FDS=False, targets=None):
        x = self.block_1(x)
        x = self.block_2(x)
        if use_FDS:
            assert targets!=None
            x = self.FDS.smooth(x,targets)
        output = self.fclayer(x)

        if return_feats:
            return output,x
        return output
    def forward_fc(self, feat):
        return self.fclayer(feat)

    def repr_forward(self, x):
        with torch.no_grad():
            x = self.block_1(x)
            repr = self.block_2(x)
            return repr
    def opim_design(self,args,rate=1.0):
        args.optimiser_args['lr'] = args.optimiser_args['lr'] * rate
        optim1 = opt.Adam([{'params': self.block_1.parameters()},
                           {'params': self.block_2.parameters()}], **args.optimiser_args)
        optim2 = opt.Adam(self.fclayer.parameters(),**args.optimiser_args)
        return optim1,optim2
    def freeze_backbone(self):
        for p in self.block_1.parameters():
            p.requires_grad = False
        for p in self.block_2.parameters():
            p.requires_grad = False
    def unfreeze_backbone(self):
        for p in self.block_1.parameters():
            p.requires_grad = True
        for p in self.block_2.parameters():
            p.requires_grad = True
    def freeze_upper(self):
        self.freeze_backbone()
        self.unfreeze_backbone()
        # for p in self.block_1.parameters():
        #     p.requires_grad = False
    
from torchvision import models

class Learner_RCF_MNIST(nn.Module):
    def __init__(self, args, weights = None):
        super(Learner_RCF_MNIST, self).__init__()
        self.args = args

        # get feature extractor from original model
        ori_model = models.resnet18(pretrained=True)
        #for param in model.parameters():
        #    param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = ori_model.fc.in_features
        # print(f'num_ftrs = {num_ftrs}')
        self.n_feat = num_ftrs
        self.IB_k = num_ftrs//2
        
        self.feature_extractor = torch.nn.Sequential(*list(ori_model.children())[:-2])
        print('--------',self.feature_extractor)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.fc = nn.Linear(num_ftrs, 1)
        self.IB_fc = nn.Linear(num_ftrs//2, 1)

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x,return_feats=False,use_FDS=False, targets=None):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        if use_FDS:
            assert targets!=None
            x = self.FDS.smooth(x,targets)
        output = self.fc(x)
        if return_feats:
            return output,x
        return output
    def forward_IB(self,x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        stat = x.flatten(1)

        mu = stat[:, :self.IB_k]
        std = F.softplus(stat[:,self.IB_k:], beta=1)

        feats = self.reparametrize_n(mu,std,1)
        logit = self.IB_fc(feats)

        return mu, std, logit
    def forward_mixup(self, x1, x2, lam = None,return_feats=False,use_FDS=False, targets=None):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)

        # mixup feature
        x = lam * x1 + (1 - lam) * x2
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if use_FDS:
            assert targets!=None
            x = self.FDS.smooth(x,targets)
        output = self.fc(x)
        if return_feats:
            return output,x
        return output
    def forward_fc(self,feat):
        return self.fc(feat)
    def opim_design(self,args,rate=1.0):
        args.optimiser_args['lr'] = args.optimiser_args['lr'] * rate
        optim1 = opt.Adam(self.feature_extractor.parameters(), **args.optimiser_args)
        optim2 = opt.Adam(self.fc.parameters(),**args.optimiser_args)
        return optim1,optim2

    def freeze_backbone(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad=False
    def unfreeze_backbone(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad=True
    def freeze_upper(self):
        self.freeze_backbone()
        for p in self.feature_extractor[-1].parameters():
            p.requires_grad=True
    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)
        eps = Variable(std.data.new(std.size()).normal_().to(std))

        return mu + eps * std




# ---> :https://github.com/laiguokun/LSTNet
class Learner_TimeSeries(nn.Module):
    def __init__(self, args, data, weights = None):
        super(Learner_TimeSeries, self).__init__()
        self.use_cuda = args.cuda
        self.P = int(args.window)
        self.m = int(data.m)
        self.hidR = int(args.hidRNN)
        self.hidC = int(args.hidCNN)
        self.hidS = int(args.hidSkip)
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.P - self.Ck)/self.skip)
        print(f'self.pt = {self.pt}')
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            print(self.hidR + self.skip * self.hidS, self.m)
            self.linear1 = nn.Linear(int(self.hidR + self.skip * self.hidS), self.m)
            self.n_feat = int(self.hidR + self.skip * self.hidS)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
            self.n_feat = self.hidR
        if (self.hw > 0): #highway -> autoregressiion
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x, return_feats=False,use_FDS=False, targets=None):
        batch_size = x.size(0)
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        if use_FDS:
            assert targets!=None
            r = self.FDS.smooth(r,targets)
        # FC
        res = self.linear1(r)
        
        #highway auto-regression
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output):
            res = self.output(res)

        if return_feats:
            return res, r
        return res

    def repr_forward(self, x):
        batch_size = x.size(0)
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        # FC
        return r
        res = self.linear1(r)
        
        #highway auto-regression
            

    def forward_mixup(self, x1, x2, lam,return_feats=False,use_FDS=False, targets=None):
        batch_size = x1.size(0)
        #CNN
        c1 = x1.view(-1, 1, self.P, self.m)
        c1 = F.relu(self.conv1(c1))
        c1 = self.dropout(c1)
        c1 = torch.squeeze(c1, 3)

        #CNN
        c2 = x2.view(-1, 1, self.P, self.m)
        c2 = F.relu(self.conv1(c2))
        c2 = self.dropout(c2)
        c2 = torch.squeeze(c2, 3)
        
        # just mixup after conv block
        c = lam * c1 + (1 - lam) * c2

        # RNN time number <-> layer number
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, int(self.pt), int(self.skip))
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(int(self.pt), int(batch_size * self.skip), int(self.hidC))
            _, s = self.GRUskip(s)
            s = s.view(batch_size, int(self.skip * self.hidS))
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        # FC
        res = self.linear1(r)
        
        #highway auto-regression --> not mixup
        if (self.hw > 0):
            x = lam * x1 + (1 - lam) * x2
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output):
            res = self.output(res)
        if return_feats:
            return res, r
        return res
    def forward_fc(self,x,feat):
        res = self.linear1(feat)
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res
    def opim_design(self,args,rate=1.0):
        args.optimiser_args['lr'] = args.optimiser_args['lr'] * rate
        params = [{'params': self.conv1.parameters()},
                  {'params': self.GRU1.parameters()}]
        if self.skip>0:
            params.append({'params':self.GRUskip.parameters()})

        optim1 = opt.Adam(params, **args.optimiser_args)
        optim2 = opt.Adam(self.linear1.parameters(),**args.optimiser_args)
        return optim1,optim2

    def freeze_backbone(self):
        for p in self.conv1.parameters():
            p.requires_grad=False
        for p in self.GRU1.parameters():
            p.requires_grad=False
        if self.skip>0:
            for p in self.GRUskip.parameters():
                p.requires_grad=False
    def unfreeze_backbone(self):
        for p in self.conv1.parameters():
            p.requires_grad=True
        for p in self.GRU1.parameters():
            p.requires_grad=True
        if self.skip>0:
            for p in self.GRUskip.parameters():
                p.requires_grad=True
    def freeze_upper(self):
        self.freeze_backbone()
        for p in self.GRU1.parameters():
            p.requires_grad=True
        if self.skip>0:
            for p in self.GRUskip.parameters():
                p.requires_grad=True


# ---> https://github.com/mims-harvard/TDC/tree/master/
class Learner_Dti_dg(nn.Module):
    def __init__(self, hparams = None, weights = None):
        super(Learner_Dti_dg, self).__init__()

        self.num_classes = 1
        self.input_shape = [(63, 100), (26, 1000)]
        self.num_domains = 6
        self.hparams = hparams

        self.featurizer = networks.DTI_Encoder()
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes,
            False)
            #self.hparams['nonlinear_classifier'])
        self.n_feat = self.featurizer.n_outputs

        #self.network = mySequential(self.featurizer, self.classifier)

        if weights != None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self,x, return_feats=False,use_FDS=False, targets=None):
        drug_num = self.input_shape[0][0] * self.input_shape[0][1]
        x_drug = x[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
        x_protein = x[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
        
        feature_out = self.featurizer.forward(x_drug,x_protein)
        if use_FDS:
            assert targets!=None
            feature_out = self.FDS.smooth(feature_out,targets)
        linear_out = self.classifier(feature_out)
        if return_feats:
            return linear_out, feature_out
        return linear_out
    def opim_design(self, args,rate=1.0):
        args.optimiser_args['lr'] = args.optimiser_args['lr'] *rate
        optim1 = opt.Adam(self.featurizer.parameters(), **args.optimiser_args)
        optim2 = opt.Adam(self.classifier.parameters(), **args.optimiser_args)
        return optim1, optim2

    def freeze_backbone(self):
        for p in self.featurizer.parameters():
            p.requires_grad=False
    def unfreeze_backbone(self):
        for p in self.featurizer.parameters():
            p.requires_grad=True
    def freeze_upper(self):
        self.freeze_backbone()
        for p in self.featurizer.predictor.parameters():
            p.requires_grad=True
    
    def repr_forward(self, x):
        with torch.no_grad():
            drug_num = self.input_shape[0][0] * self.input_shape[0][1]
            x_drug = x[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
            x_protein = x[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
            
            repr = self.featurizer.forward(x_drug,x_protein)
            return repr

    def forward_mixup(self,x1, x2, lambd,return_feats=False,use_FDS=False, targets=None):
        drug_num = self.input_shape[0][0] * self.input_shape[0][1]
        x1_drug = x1[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
        x1_protein = x1[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])
        
        x2_drug = x2[:,:drug_num].reshape(-1,self.input_shape[0][0],self.input_shape[0][1])
        x2_protein = x2[:,drug_num:].reshape(-1,self.input_shape[1][0],self.input_shape[1][1])

        feature_out = self.featurizer.forward_mixup(x1_drug,x1_protein,x2_drug,x2_protein,lambd)
        if use_FDS:
            assert targets!=None
            feature_out = self.FDS.smooth(feature_out,targets)
        linear_out = self.classifier(feature_out)
        if return_feats:
            return linear_out, feature_out
        return linear_out
        #return self.network.forward_mixup(x1_drug, x1_protein,x2_drug, x2_protein,lambd)
    def forward_fc(self,feat):
        return self.classifier(feat)

class Discriminator(torch.nn.Module):
    def __init__(self, input_shape):
        '''
        input shape is channel shape
        '''
        super(Discriminator, self).__init__()
        # self.conv = nn.Conv2d(input_shape,input_shape*4,kernel_size=(1,2))
        self.linear_layers = nn.Linear(input_shape,1)
        self.linear_layers2 = nn.Sequential(
            nn.Linear(input_shape+1,(input_shape+1)//2),
            # # nn.ReLU(),
            # # nn.BatchNorm1d(input_shape*2),
            nn.Linear((input_shape+1)//2,1),
            # nn.ReLU(),
            # # nn.BatchNorm1d(input_shape),
            # nn.Linear(input_shape,1),
            # GradientReversalLayer(alpha=1)
        )
    def forward(self,x):
        # x = self.conv(x)
        # x = x.squeeze()
        x = self.linear_layers(x)
        x = x.squeeze().unsqueeze(0)
        x = self.linear_layers2(x)
        return x


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply
class GradientReversalLayer(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)

