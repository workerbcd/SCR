import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import copy
import random
import json

import src.DDG.DDG_network as networks
import src.DDG.hparam as hparams_registry
import lib.mics as misc

def get_hparams(args):

    hparams = hparams_registry.default_hparams('DDG', args.dataset)


    # if args.hparams:
    #     hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
    return hparams
class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains,args, device='cpu'):
        super(Algorithm, self).__init__()
        self.hparams = get_hparams(args)
        self.device = device

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self,  input_shape, num_classes, num_domains,args, device='cpu'):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,args,
                                   device)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # self.model = model

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class DDG(ERM):
    def __init__(self,input_shape, num_classes, num_domains,args, device='cpu'):
        super(DDG, self).__init__(input_shape, num_classes, num_domains,args,
                                  device)
        self.iteration = 0
        self.id_featurizer = self.featurizer
        self.dis_id = self.classifier
        self.gen = networks.AdaINGen(1, self.id_featurizer.n_outputs, self.hparams) if not self.hparams[
            'is_mnist'] else networks.VAEGen()
        self.dis_img = networks.MsImageDis(hparams=self.hparams)
        self.recon_xp_w = self.hparams['recon_xp_w']
        self.recon_xn_w = self.hparams['recon_xn_w']
        self.margin = self.hparams['margin']
        self.eta = self.hparams['eta']

        def to_gray(half=False):  # simple
            def forward(x):
                x = torch.mean(x, dim=1, keepdim=True)
                if half:
                    x = x.half()
                return x

            return forward

        self.single = to_gray(False)
        self.optimizer_gen = torch.optim.Adam([p for p in list(self.gen.parameters()) if p.requires_grad],
                                              lr=self.hparams['lr_g'], betas=(0, 0.999),
                                              weight_decay=self.hparams['weight_decay_g'])
        if self.hparams['stage'] == 0:
            # Setup the optimizers
            self.optimizer_dis_img = torch.optim.Adam(
                self.dis_img.parameters(),
                lr=self.hparams["lr_d"],
                weight_decay=self.hparams['weight_decay'])
            step = self.hparams['steps'] * 0.6
            print(step)
            self.dis_scheduler = lr_scheduler.MultiStepLR(self.optimizer_dis_img, milestones=[step, step + step // 2,
                                                                                              step + step // 2 + step // 4],
                                                          gamma=0.1)
            self.gen_scheduler = lr_scheduler.MultiStepLR(self.optimizer_gen, milestones=[step, step + step // 2,
                                                                                          step + step // 2 + step // 4],
                                                          gamma=0.1)

        self.id_criterion = nn.MSELoss()
        self.dom_criterion = nn.MSELoss()

    def recon_criterion(self, input, target, reduction=True):
        diff = input - target.detach()
        B, C, H, W = input.shape
        if reduction == False:
            return torch.mean(torch.abs(diff[:]).view(B, -1), dim=-1)
        return torch.mean(torch.abs(diff[:]))

    def train_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
            print('there has bn')

    def forward(self, x_a, x_b, xp_a, xp_b):
        '''
            inpus:
                x_a, x_b: image from dataloader a,b
                xp_a, xp_b: positive pair of x_a, x_b
        '''
        s_a = self.gen.encode(self.single(x_a))  # v for x_a
        s_b = self.gen.encode(self.single(x_b))  # v for x_b
        f_a, x_fa = self.id_featurizer(x_a, self.hparams['stage'])  # f_a: detached s for x_a, x_fa: s for x_a
        p_a = self.dis_id(x_fa)  # identity classification result for x_a
        f_b, x_fb = self.id_featurizer(x_b, self.hparams['stage'])
        p_b = self.dis_id(x_fb)
        fp_a, xp_fa = self.id_featurizer(xp_a, self.hparams['stage'])
        pp_a = self.dis_id(xp_fa)
        fp_b, xp_fb = self.id_featurizer(xp_b, self.hparams['stage'])
        pp_b = self.dis_id(xp_fb)
        # if self.hparams['stage'] == 0:
        #     # cross-style generation
        #     x_ba = self.gen.decode(s_b, f_a) # x_ba: generated from identity of a and style of b
        #     x_ab = self.gen.decode(s_a, f_b)
        #     x_a_recon = self.gen.decode(s_a, f_a) # generate from identity and style of a
        #     x_b_recon = self.gen.decode(s_b, f_b)
        # else:
        #     x_ba = None
        #     x_ab = None
        #     x_a_recon = None
        #     x_b_recon = None
        # cross-style generation
        x_ba = self.gen.decode(s_b, f_a)  # x_ba: generated from identity of a and style of b
        x_ab = self.gen.decode(s_a, f_b)
        x_a_recon = self.gen.decode(s_a, f_a)  # generate from identity and style of a
        x_b_recon = self.gen.decode(s_b, f_b)
        x_a_recon_p = self.gen.decode(s_a, fp_a)
        x_b_recon_p = self.gen.decode(s_b, fp_b)

        return x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p

    def dis_update(self, x_ab, x_ba, x_a, x_b, hparams):
        '''
            inpus:
                x_ab: generated from identity of b and style of a (fake)
                x_ba: generated from identity of a and style of b (fake)
                x_a, x_b: real image
        '''
        self.optimizer_dis_img.zero_grad()
        self.loss_dis_a, reg_a = self.dis_img.calc_dis_loss(self.dis_img, x_ba.detach(), x_a)
        self.loss_dis_b, reg_b = self.dis_img.calc_dis_loss(self.dis_img, x_ab.detach(), x_b)
        self.loss_dis_total = hparams['gan_w'] * self.loss_dis_a + hparams['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()  # discriminators are trained here
        self.optimizer_dis_img.step()

    def gen_update(self, x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p,
                   x_b_recon_p, x_a, x_b, l_a, l_b, hparams):
        '''
            inputs:
                x_ab: generated from identity of b and style of a
                x_ba: generated from identity of a and style of b
                s_a, s_b: style factors for x_a, x_b
                f_a, f_b: detached semantic factors for x_a, x_b
                p_a, p_b: identity prediction results for x_a, x_b
                pp_a, pp_b: identity prediction results for the positive pair of x_a, x_b
                x_a_recon, x_b_recon: reconstruction of x_a, x_b
                x_a_recon_p, x_b_recon_p: reconstruction of the positive pair of x_a, x_b
                x_a, x_b,  l_a, l_b: images and identity labels
                hparams: parameters
        '''
        self.optimizer_gen.zero_grad()
        self.optimizer.zero_grad()

        #################################

        # auto-encoder image reconstruction
        self.recon_a2a, self.recon_b2b = self.recon_criterion(x_a_recon_p, x_a, reduction=False), self.recon_criterion(
            x_b_recon_p, x_b, reduction=False)
        self.loss_gen_recon_p = torch.mean(
            torch.max(self.recon_a2a - self.margin, torch.zeros_like(self.recon_a2a))) + torch.mean(
            torch.max(self.recon_b2b - self.margin, torch.zeros_like(self.recon_b2b)))

        # Emprical Loss
        if not hparams['is_mnist']:
            _, x_fa_recon = self.id_featurizer(x_ab)
            p_a_recon = self.dis_id(x_fa_recon)
            _, x_fb_recon = self.id_featurizer(x_ba)
            p_b_recon = self.dis_id(x_fb_recon)
        else:
            _, x_fa_recon = self.id_featurizer(x_ba)
            p_a_recon = self.dis_id(x_fa_recon)
            _, x_fb_recon = self.id_featurizer(x_ab)
            p_b_recon = self.dis_id(x_fb_recon)
        self.loss_id = self.id_criterion(p_a, l_a) + self.id_criterion(p_b, l_b) + self.id_criterion(pp_a,
                                                                                                     l_a) + self.id_criterion(
            pp_b, l_b)
        self.loss_gen_recon_id = self.id_criterion(p_a_recon, l_a) + self.id_criterion(p_b_recon, l_b)

        self.step(torch.mean(self.recon_a2a))
        # total loss
        self.loss_gen_total = self.loss_id + \
                              self.recon_xp_w * self.loss_gen_recon_p + \
                              hparams['recon_id_w'] * self.loss_gen_recon_id

        self.loss_gen_total.backward()
        self.optimizer_gen.step()
        self.optimizer.step()

    def gan_update(self, x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p,
                   x_b_recon_p, x_a, x_b, l_a, l_b, hparams):
        '''
            Train the GAN
            inputs:
                x_ab: generated from identity of b and style of a
                x_ba: generated from identity of a and style of b
                s_a, s_b: style factors for x_a, x_b
                f_a, f_b: detached semantic factors for x_a, x_b
                p_a, p_b: identity prediction results for x_a, x_b
                pp_a, pp_b: identity prediction results for the positive pair of x_a, x_b
                x_a_recon, x_b_recon: reconstruction of x_a, x_b
                x_a_recon_p, x_b_recon_p: reconstruction of the positive pair of x_a, x_b
                x_a, x_b,  l_a, l_b: images and identity labels
                hparams: parameters
        '''
        self.optimizer_gen.zero_grad()
        self.optimizer.zero_grad()

        # no gradient
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)
        f_a, f_b = f_a.detach(), f_b.detach()

        rand_num = random.uniform(0, 1)
        #################################
        # encode structure
        if 0.5 >= rand_num:
            # encode again (encoder is tuned, input is fixed)
            s_a_recon = self.gen.enc_content(self.single(x_ab_copy))
            s_b_recon = self.gen.enc_content(self.single(x_ba_copy))
        else:
            # copy the encoder
            self.enc_content_copy = copy.deepcopy(self.gen.enc_content)
            self.enc_content_copy = self.enc_content_copy.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = self.enc_content_copy(self.single(x_ab))
            s_b_recon = self.enc_content_copy(self.single(x_ba))

        #################################
        # encode appearance
        self.id_copy = copy.deepcopy(self.id_featurizer)
        self.dis_id_copy = copy.deepcopy(self.dis_id)
        self.id_copy.eval()
        self.dis_id_copy.eval()

        # encode again (encoder is fixed, input is tuned)
        f_a_recon, _ = self.id_copy(x_ba)
        f_b_recon, _ = self.id_copy(x_ab)

        # auto-encoder image reconstruction
        self.loss_gen_recon_x = self.recon_criterion(x_a_recon, x_a) + self.recon_criterion(x_b_recon, x_b)

        # Emprical Loss

        x_aba, x_bab = self.gen.decode(s_a_recon, f_a_recon), self.gen.decode(s_b_recon, f_b_recon) if hparams[
                                                                                                           'recon_x_cyc_w'] > 0 else None
        self.loss_gen_cycrecon_x = self.recon_criterion(x_aba, x_a) + self.recon_criterion(x_bab, x_b) if hparams[
                                                                                                              'recon_x_cyc_w'] > 0 else torch.tensor(
            0)

        # GAN loss
        self.loss_gen_adv = self.dis_img.calc_gen_loss(self.dis_img, x_ba) + self.dis_img.calc_gen_loss(self.dis_img,
                                                                                                        x_ab)

        self.step()
        if self.iteration > hparams['steps'] * hparams['warm_iter_r']:
            hparams['recon_x_cyc_w'] += hparams['warm_scale']
            hparams['recon_x_cyc_w'] = min(hparams['recon_x_cyc_w'], hparams['max_cyc_w'])

        # total loss
        self.loss_gen_total = hparams['gan_w'] * self.loss_gen_adv + \
                              hparams['recon_x_w'] * self.loss_gen_recon_x + \
                              hparams['recon_x_cyc_w'] * self.loss_gen_cycrecon_x

        self.loss_gen_total.backward()
        self.optimizer_gen.step()
        self.optimizer.step()

    def update(self, minibatches, minibatches_neg, pretrain_model=None, unlabeled=None, iteration=0):
        images_a = torch.cat([x for x, y, pos in minibatches])
        labels_a = torch.cat([y for x, y, pos in minibatches])
        pos_a = torch.cat([pos for x, y, pos in minibatches])
        images_b = torch.cat([x for x, y, pos in minibatches_neg])
        labels_b = torch.cat([y for x, y, pos in minibatches_neg])
        pos_b = torch.cat([pos for x, y, pos in minibatches_neg])

        if self.hparams['stage'] == 1 and pretrain_model is not None:
            # swap semantic factors
            s_a = pretrain_model.gen.encode(self.single(images_a))  # v for x_a
            s_b = pretrain_model.gen.encode(self.single(images_b))  # v for x_b
            f_a, x_fa = pretrain_model.id_featurizer(images_a)  # f_a: detached s for x_a, x_fa: s for x_a
            f_b, x_fb = pretrain_model.id_featurizer(images_b)
            # cross-style generation
            x_ba = pretrain_model.gen.decode(s_b, f_a)  # x_ba: generated from identity of a and style of b
            x_ab = pretrain_model.gen.decode(s_a, f_b)
            _, _, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = self.forward(
                images_a, images_b, pos_a, pos_b)
        else:
            x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = self.forward(
                images_a, images_b, pos_a, pos_b)

        if self.hparams['stage'] == 0:
            self.dis_update(x_ab.clone(), x_ba.clone(), images_a, images_b, self.hparams)
            self.gan_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p,
                            x_b_recon_p, images_a, images_b, labels_a, labels_b, self.hparams)
            self.gen_scheduler.step()
            self.dis_scheduler.step()
            return {'loss_total': self.loss_gen_total.item(),
                    'loss_gan': self.loss_gen_adv.item(),
                    'loss_recon_x': self.loss_gen_recon_x.item(),
                    'loss_x_cyc': self.loss_gen_cycrecon_x.item()}
        else:
            self.gen_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p,
                            x_b_recon_p, images_a, images_b, labels_a, labels_b, self.hparams)
            return {
                'loss_cls': self.loss_id.item(),
                'loss_gen_recon_id': self.loss_gen_recon_id.item(),
                'recon_xp_w': self.recon_xp_w,
                'loss_recon_p': self.loss_gen_recon_p.item()}

    def sample(self, x_a_, x_b_, pretrain_model=None):
        self.eval()
        x_a = torch.cat([x for x, y, pos in x_a_])
        x_b = torch.cat([x for x, y, pos in x_b_])
        bn = x_b.size(0)
        perm = torch.randperm(bn)
        x_b = x_b[perm]
        x_a_recon, x_b_recon, x_ba1, x_ab1, x_aba, x_bab = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            model = pretrain_model if pretrain_model is not None else self
            s_a = model.gen.encode(self.single(x_a[i].unsqueeze(0)))
            s_b = model.gen.encode(self.single(x_b[i].unsqueeze(0)))
            f_a, _ = model.id_featurizer(x_a[i].unsqueeze(0))
            f_b, _ = model.id_featurizer(x_b[i].unsqueeze(0))
            x_a_recon.append(model.gen.decode(s_a, f_a))
            x_b_recon.append(model.gen.decode(s_b, f_b))
            x_ba = model.gen.decode(s_b, f_a)
            x_ab = model.gen.decode(s_a, f_b)
            x_ba1.append(x_ba)
            x_ab1.append(x_ab)

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ab1 = torch.cat(x_ba1), torch.cat(x_ab1)
        self.train()

        return x_a, x_ba1, x_b, x_ab1

    def predict(self, x):
        return self.dis_id(self.id_featurizer(x)[-1])

    def step(self, recon_p=None):
        self.iteration += 1
        if recon_p is None:
            return
        self.recon_xp_w = min(max(self.recon_xp_w + self.eta * (recon_p.item() - self.margin), 0), 1)