import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import model
import transform as tran
import numpy as np
import os
import argparse
import copy
from datetime import datetime
import time
# torch.set_num_threads(1)
from read_data import ImageList
import shutil
from cmixup import get_batch_kde_mixup_batch
from losses.RML import RegMetricLoss
from losses.myloss import std_svd2


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch DARE-GRAM experiment')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0',
                    help="device id to run")
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--src', type=str, default='c', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='s', metavar='S',
                    help='target dataset')
parser.add_argument('--lr', type=float, default=0.1,
                    help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001,
                    help='learning rate decay')
parser.add_argument('--kde_bandwidth', type=float, default=0.5,
                    help='bandwidth')
parser.add_argument('--kde_type', type=str, default='gaussian',
                    help='kde_type')
parser.add_argument('--treshold', type=float, default=0.9,
                    help='treshold for the pseudo inverse')
parser.add_argument('--batch', type=int, default=36,
                    help='batch size')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--svd_w', type=float,default=1.0)
parser.add_argument('--std_w', type=float,default=1.0)
args = parser.parse_args()
def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_transforms = {
    'train': tran.rr_train(resize_size=224),
    'val': tran.rr_train(resize_size=224),
    'test': tran.rr_eval(resize_size=224),
}
# set dataset
batch_size = {"train": args.batch, "val": args.batch, "test": 36}
rc = "realistic.txt"
rl = "real.txt"
t = "toy.txt"

rc_t = "realistic_test.txt"
rl_t = "real_test.txt"
t_t = "toy_test.txt"

# if args.src == 'rl':
#     source_path = rl
# elif args.src == 'rc':
#     source_path = rc
# elif args.src == 't':
#     source_path = t
#
# if args.tgt == 'rl':
#     target_path = rl
# elif args.tgt == 'rc':
#     target_path = rc
# elif args.tgt == 't':
#     target_path = t

if args.tgt == 'rl':
    target_path_t = rl_t
    source_path1 = rc
    source_path2 = t
    source_path1_t = rc_t
    source_path2_t = t_t
elif args.tgt == 'rc':
    target_path_t = rc_t
    source_path1 = rl
    source_path2 = t
    source_path1_t = rl_t
    source_path2_t = t_t
elif args.tgt == 't':
    target_path_t = t_t
    source_path1 = rl
    source_path2 = t
    source_path1_t = rl_t
    source_path2_t = t_t

dsets = {"train": ImageList(open(source_path1).readlines()+open(source_path2).readlines(), transform=data_transforms["train"]), 
         "val": ImageList(open(source_path1_t).readlines()+open(source_path2_t).readlines(),transform=data_transforms["val"]),
         "test": ImageList(open(target_path_t).readlines(),transform=data_transforms["test"])}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=4)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log_root = '/home/s4686009/remotedata/MPI3D/logs'
global_stamp = timestamp()
log_path = os.path.join(log_root, f'{global_stamp}_{args.name}_{args.tgt}_svdw{args.svd_w}_stdw{args.std_w}_{args.seed}')
if not os.path.exists(log_path):
    os.makedirs(log_path)
print(log_path)
shutil.copytree('.', os.path.join(log_path, 'src_mdi3d') )
def Regression_test(loader, model):
    MSE = [0, 0, 0]
    MAE = [0, 0, 0]
    number = 0
    stamp = timestamp()
    log_name = os.path.join(log_path,f'{stamp}_test.txt')
    with torch.no_grad():
        for (imgs, labels, _) in loader['test']:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels1 = labels[:, 0]
            labels2 = labels[:, 1]
            labels1 = labels1.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
            labels = torch.cat((labels1, labels2), dim=1)
            labels = labels.float() / 39
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(3):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print("\tMSE : {0},{1}\n".format(MSE[0], MSE[1]))
    print("\tMAE : {0},{1}\n".format(MAE[0], MAE[1]))
    print("\tMSEall : {0}\n".format(MSE[2]))
    print("\tMAEall : {0}\n".format(MAE[2]))
    with open(log_name, 'a+') as f:
        f.write("MSE : {0},{1}".format(MSE[0], MSE[1]) + '\n')
        f.write("MAE : {0},{1}".format(MAE[0], MAE[1]) + '\n')
        f.write("MSEall : {0}".format(MSE[2]) + '\n')
        f.write("MAEall: {0}".format(MAE[2]) + '\n')
    return MAE[2]

def Regression_val(loader, model):
    MSE = [0, 0, 0]
    MAE = [0, 0, 0]
    number = 0
    stamp = timestamp()
    log_name = os.path.join(log_path, f'{stamp}_val.txt')
    with torch.no_grad():
        for (imgs, labels, _) in loader['val']:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels1 = labels[:, 0]
            labels2 = labels[:, 1]
            labels1 = labels1.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
            labels = torch.cat((labels1, labels2), dim=1)
            labels = labels.float() / 39
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(3):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print("\tMSE : {0},{1}\n".format(MSE[0], MSE[1]))
    print("\tMAE : {0},{1}\n".format(MAE[0], MAE[1]))
    print("\tMSEall : {0}\n".format(MSE[2]))
    print("\tMAEall : {0}\n".format(MAE[2]))
    with open(log_name, 'a+') as f:
        f.write("MSE : {0},{1}".format(MSE[0], MSE[1]) + '\n')
        f.write("MAE : {0},{1}".format(MAE[0], MAE[1]) + '\n')
        f.write("MSEall : {0}".format(MSE[2]) + '\n')
        f.write("MAEall: {0}".format(MSE[2]) + '\n')
    return MSE[2], MAE[2]


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay
        i += 1
    return optimizer


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression, self).__init__()
        self.model_fc = model.Resnet18Fc()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer = nn.Sequential(self.classifier_layer, nn.Sigmoid())
        self.predict_layer = nn.Sequential(self.model_fc, self.classifier_layer)

    def forward(self, x,target=None):
        feature = self.model_fc(x)
        if target is not None:
            feature = self.FDS.smooth(x,target)
        outC = self.classifier_layer(feature)

        return outC, feature
    def freeze_upper(self):
        self.model_fc.freeze_upper()


Model_R = Model_Regression()
Model_R = Model_R.to(device)

Model_R.train(True)
criterion = {"regressor": nn.MSELoss(), "rml": RegMetricLoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, Model_R.classifier_layer.parameters()), "lr": 1}]

optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)

len_source = len(dset_loaders["train"]) - 1
len_target = len(dset_loaders["val"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])

for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])
print_interval = 500
test_interval = 1000
num_iter = 5000

print(args)
def train_erm(optimizer,iter_train,iter_val):
    train_regression_loss = train_loss_svd = train_loss_std = train_total_loss = 0.0
    best_model = copy.deepcopy(Model_R)
    best_mse = 1e12
    for iter_num in range(1, num_iter + 1):
        Model_R.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                     weight_decay=0.0005)
        optimizer.zero_grad()
        if iter_num % len_source == 0:
            iter_train = iter(dset_loaders["train"])

        data_train = iter_train.next()

        inputs_train, labels_train, index_train = data_train
        start = time.perf_counter()

        labels1 = labels_train[:, 0]
        labels2 = labels_train[:, 1]
        labels1 = labels1.unsqueeze(1)
        labels2 = labels2.unsqueeze(1)

        labels_train = torch.cat((labels1, labels2), dim=1)
        labels_train = labels_train.float()/39


        inputs = inputs_train
        inputs = inputs.to(device)
        labels = labels_train.to(device)

        end = time.perf_counter()


        outC, feature = Model_R(inputs)

        regression_loss =criterion["regressor"](outC, labels)

        total_loss = regression_loss

        total_loss.backward()
        optimizer.step()

        train_regression_loss += regression_loss.item()
        train_total_loss += total_loss.item()

        if iter_num % print_interval == 0:
            print(
                "Iter {:05d}, Average MSE Loss: {:.4f}; Average std Loss: {:.4f};Average svd Loss: {:.4f};Average Training Loss: {:.4f}".format(
                    iter_num, train_regression_loss / float(print_interval), train_loss_std / float(print_interval), train_loss_svd / float(print_interval),train_total_loss / float(print_interval)))
            train_regression_loss = train_loss_std=train_loss_svd=train_total_loss = 0.0
            print("time is {}".format(end-start))
        if (iter_num % test_interval) == 0:
            Model_R.eval()
            mse, mae = Regression_val(dset_loaders, Model_R.predict_layer)
            if best_mse > mse:
                best_model = copy.deepcopy(Model_R)
                best_mse = mse
    return best_model

def train_mixup(optimizer,iter_train,iter_val):
    if 'svd' in args.name:
        print("svd loaded")
    if 'std' in args.name:
        print("std loaded")
    train_regression_loss = train_loss_svd = train_loss_std = train_total_loss = 0.0
    trainset = dsets["train"]
    sample_pool = np.arange(len(trainset))
    best_model = copy.deepcopy(Model_R)
    best_mse = 1e12
    for iter_num in range(1, num_iter + 1):
        lambd = np.random.beta(2.0, 2.0)
        Model_R.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                     weight_decay=0.0005)
        optimizer.zero_grad()
        if iter_num*2 % len_source == 0:
            iter_train = iter(dset_loaders["train"])

        data_train = iter_train.next()
        data_train2 = iter_train.next()

        inputs_train, labels_train, index_train = data_train
        X2, labels_train2, _ =data_train2
        start = time.perf_counter()

        labels1 = labels_train[:, 0]
        labels2 = labels_train[:, 1]
        labels1 = labels1.unsqueeze(1)
        labels2 = labels2.unsqueeze(1)

        labels1_ = labels_train2[:, 0]
        labels2_ = labels_train2[:, 1]
        labels1_ = labels1_.unsqueeze(1)
        labels2_ = labels2_.unsqueeze(1)

        labels_train = torch.cat((labels1, labels2), dim=1)
        labels_train = labels_train.float()/39
        labels_train_ = torch.cat((labels1_, labels2_), dim=1)
        Y2 = labels_train_.float() / 39

        inputs = inputs_train
        X2 = X2.to(device)
        Y2 = Y2.to(device)

        inputs = inputs.to(device)
        labels = labels_train.to(device)

        X2, Y2 = get_batch_kde_mixup_batch(args, inputs, X2, labels, Y2, device)
        end = time.perf_counter()
        mixup_Y = labels * lambd + Y2 * (1 - lambd)
        mixup_X = inputs * lambd + X2 * (1 - lambd)

        outC, feature = Model_R(inputs)
        # outC_t, feature_t = Model_R(inputs_t)
        out_mixup, feature_mixup = Model_R(mixup_X)

        regression_loss =criterion["regressor"](outC, labels)+criterion["regressor"](out_mixup, mixup_Y)
        std1 = std_svd2(feature,labels)
        std2 = std_svd2(feature_mixup,mixup_Y)
        loss_std = std1+std2
        _,svd1,_ = torch.linalg.svd(feature)
        _,svd2,_ = torch.linalg.svd(feature_mixup)
        loss_svd = torch.abs(svd1[0]-svd2[0])
        total_loss = regression_loss
        # +args.std_w * loss_std#+ args.svd_w*loss_svd#loss_ranksim #+ rml_loss
        if 'svd' in args.name:
            total_loss += args.svd_w*loss_svd
        if 'std' in args.name:
            total_loss += args.std_w * loss_std

        total_loss.backward()
        optimizer.step()

        train_regression_loss += regression_loss.item()
        train_total_loss += total_loss.item()

        if iter_num % print_interval == 0:
            print(
                "Iter {:05d}, Average MSE Loss: {:.4f}; Average std Loss: {:.4f};Average svd Loss: {:.4f};Average Training Loss: {:.4f}".format(
                    iter_num, train_regression_loss / float(print_interval), train_loss_std / float(print_interval), train_loss_svd / float(print_interval),train_total_loss / float(print_interval)))
            train_regression_loss = train_loss_std=train_loss_svd=train_total_loss = 0.0
            print("time is {}".format(end-start))
        if (iter_num % test_interval) == 0:
            Model_R.eval()
            mse, mae = Regression_val(dset_loaders, Model_R.predict_layer)
            if best_mse > mse:
                best_model = copy.deepcopy(Model_R)
                best_mse = mse
    return best_model


if __name__ == '__main__':
    if args.name == 'erm':
        best_model = train_erm(optimizer,iter_train=iter_source,iter_val=iter_target)
    else:
        best_model = train_mixup(optimizer,iter_train=iter_source, iter_val=iter_target)
    # torch.save(best_model, os.path.join(log_path, 'best_model.pth'))
    print('--------------final evaluation----------------')
    best_model.eval()
    Regression_test(dset_loaders, best_model.predict_layer)


