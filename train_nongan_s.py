# coding:utf8
# 基础版是train_GBU_CC_.py,目前结果最好的版本
# 毕设将cross_gan改为对抗式和非对抗式
# 本脚本为非对抗式
# 在 nongan 版本的基础上，在 loss 里加上 dissimilarity 的约束
# 效果没有改进，弃用
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn


from termcolor import cprint
from time import gmtime, strftime
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import argparse
import os
import glob
import random
import json
import h5py
import dateutil.tz
import datetime

from tensorboardX import SummaryWriter

from dataset import FeatDataLayer, DATA_LOADER, KnnFeat, SVM, Softmax
from models import _netCC_3, _param

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2')
parser.add_argument('--dataroot', default='/data/liujinlu/xian_resnet101/xlsa17/data/',
                    help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')

parser.add_argument('--gpu', default='3', type=str, help='index of GPU to use')
parser.add_argument('--exp_idx', default='', type=str, help='exp idx')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume',  type=str, help='the model to resume')

parser.add_argument('--z_dim',  type=int, default=300, help='dimension of the random vector z')
parser.add_argument('--disp_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=300)
parser.add_argument('--evl_interval',  type=int, default=40)

parser.add_argument('--random', action='store_true', default=False, help = 'random pairs in data preparation')
parser.add_argument('--endEpoch', type = int, default = 5000, help= 'train epoch')
parser.add_argument('--gzsl', action='store_true', default = False, help = 'gzsl evaluation')
parser.add_argument('--batchsize', type = int, default = 1024, help= 'batchsize')
parser.add_argument('--k_class', type = int, default = 1, help= 'find k similar classes')
parser.add_argument('--k_inst', type = int, default = 1, help= 'find k similar instances in each similar class')
parser.add_argument('--att_w', type = int, default = 10, help= 'weight of Att_loss')
parser.add_argument('--x_w', type = int, default = 40, help= 'weight of X_loss')


opt = parser.parse_args()
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.GP_LAMBDA = 10    # Gradient penalty lambda
opt.CENT_LAMBDA  = 5
opt.REG_W_LAMBDA = 0.001
opt.Adv_LAMBDA = 1

opt.lr = 0.0001

""" hyper-parameter for testing"""
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20      # knn: the value of K
opt.in_features = 600
opt.out_features = 2048
opt.G_epoch = 1
opt.D_epoch = 1
opt.t = 0.01 # 计算adj: e**(-np.linalg.norm(s1-s2)/t)
opt.path_root = '/data/liujinlu/zsl/Cross_Class_GAN/data/%s' %opt.dataset



if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)



def train():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    out_dir  = opt.path_root + '/Nongan_S'
    if not opt.random:
        out_subdir = opt.path_root + '/Nongan_S/Kcla{:d}_Kinst{:d}_{:s}'.format(opt.k_class,opt.k_inst,timestamp)
    else:
        out_subdir = opt.path_root + '/Nongan_S/Random_{:s}'.format(timestamp)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir = out_subdir + '/log.txt'
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
        for k, v in zip(vars(opt).keys(),vars(opt).values()):
            f.write(k+':'+str(v))
    summary_writer = SummaryWriter(out_subdir)


    Tensor = torch.cuda.FloatTensor
    param = _param()
    dataset = DATA_LOADER(opt)
    preinputs = KnnFeat(opt)
    param.X_dim = dataset.feature_dim
    if opt.random:
        data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.numpy(), preinputs.train_sim_feature.numpy(), preinputs.train_sim_att_1.numpy(), preinputs.train_sim_att_2.numpy(), opt)
    else:
        data_layer = FeatDataLayer(preinputs.train_sim_label.numpy(), preinputs.train_sim_output.numpy(), preinputs.train_sim_feature.numpy(), preinputs.train_sim_att_1.numpy(), preinputs.train_sim_att_2.numpy(), opt)
    
    result = Result()
    result_gzsl = Result()

    netG = _netCC_3(preinputs.train_sim_att_2.numpy().shape[1]).cuda()

    start_step = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    nets = [netG]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).cuda()
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    BCELoss = nn.BCELoss()

    for it in range(start_step, opt.endEpoch):
        print('epoch: ',it)

        """ Generator """
        for _ in range(opt.G_epoch):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # target class sample
            x = blobs['x'] # source class sample
            att = blobs['att2'] # target class attribute
            labels = blobs['labels'].astype(int)  # target class labels

            x = Variable(torch.from_numpy(x.astype('float32'))).cuda()
            att = Variable(torch.from_numpy(att.astype('float32'))).cuda()

            valid = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)

            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()

            # 
            G_sample, G_att = netG(x, att)

            # GAN's att loss
            Att_loss = opt.att_w * F.mse_loss(G_att, att)

            # GAN's x loss
            X_loss = opt.x_w * F.mse_loss(G_sample, X)


            # dissimilarity loss
            Diss_loss = Variable(torch.Tensor([0.0])).cuda()

            fake_dist = compute_dissimilarity(G_sample, tr_cls_centroid)
            real_dist = compute_dissimilarity(X, tr_cls_centroid)

            for i in range(dataset.train_cls_num):
                sample_idx = (y_true == i).data.nonzero().squeeze()
                if sample_idx.numel() == 0:
                    Diss_loss += 0.0
                else:
                    fake_diss = fake_dist[sample_idx, :]
                    real_diss = real_dist[sample_idx, :]
                    Diss_loss += (fake_diss.mean(dim=0) - real_diss.mean(dim=0)).pow(2).sum().sqrt()
            Diss_loss *= 1.0/dataset.train_cls_num * opt.CENT_LAMBDA

            # # Centroid loss
            Euclidean_loss_target = Variable(torch.Tensor([0.0])).cuda()
            # center loss
            if opt.REG_W_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss_target += 0.0

                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        Euclidean_loss_target += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
                        
                Euclidean_loss_target *= 1.0/dataset.train_cls_num * opt.CENT_LAMBDA
                

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)
            

            all_loss = reg_loss + Att_loss + X_loss + Euclidean_loss_target + Diss_loss
            all_loss.backward()

            optimizerG.step()
            summary_writer.add_scalar('G_loss', all_loss.item(), it)
            reset_grad(nets)


        if it % opt.disp_interval == 0 and it:

            log_text = 'Iter-{}; Euclidean_loss_target:{:.3f};reg_loss:{:.3f}; att_loss:{:.3f}; x_loss:{:.3f}; Diss_loss:{:.3f}'.format(it, Euclidean_loss_target.data[0],reg_loss.data[0],Att_loss.data[0], X_loss.data[0], Diss_loss.data[0])
            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text+'\n')

        if it % opt.evl_interval == 0 and it >= 40:
            netG.eval()
            eval_fakefeat_test(it, netG, preinputs, opt, result, dataset)
            if result.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model_ZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                
                save_model(it, netG, opt.manualSeed, log_text,
                           out_subdir + '/Best_model_ZSL_Acc_{:.2f}_Epoch_{:d}.tar'.format(result.acc_list[-1],it))
            if opt.gzsl:
                eval_fakefeat_test_gzsl(it, netG, preinputs, opt, result_gzsl, dataset)
                if result_gzsl.save_model:
                    files2remove = glob.glob(out_subdir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    
                    save_model(it, netG, opt.manualSeed, log_text,
                               out_subdir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}_Epoch_{:d}.tar'.format(result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T,it))
                
            netG.train()

    summary_writer.close()


def save_model(it, netG, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


def eval_fakefeat_test(it, netG, preinputs, opt, result, dataset):
    test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label = preinputs.test_sim_feature.numpy(), preinputs.test_sim_att_1.numpy(), preinputs.test_sim_att_2.numpy(), preinputs.test_sim_label.numpy()
    cur = 0
    G_sample = np.zeros([0, 2048])
    while(True):
        if cur+opt.batchsize >= len(test_sim_feature):
            temp_x = test_sim_feature[cur:cur+opt.batchsize]
            temp_att = test_sim_att_2[cur:cur+opt.batchsize]
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            g_sample, _ = netG(temp_x, temp_att)
            G_sample = np.vstack((G_sample, g_sample.numpy()))
            cur = cur + opt.batchsize
        else:
            temp_x = test_sim_feature[cur:]
            temp_att = test_sim_att_2[cur:]
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            g_sample, _ = netG(temp_x, temp_att)
            G_sample = np.vstack((G_sample, g_sample.detach().cpu().numpy()))
            break
    
    unseen_label = []
    for i in dataset.test_unseen_label.numpy():
        unseen_label.append(i+dataset.ntrain_class)
    unseen_label = np.array(unseen_label)

    # classifier = SVM(G_sample, test_sim_label)
    classifier = Softmax(G_sample, test_sim_label)
    acc = classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True
    print("Accuracy is {:.2f}%".format(acc))


def eval_fakefeat_test_gzsl(it, netG, preinputs, opt, result, dataset):
    # test_unseen
    test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label = preinputs.test_sim_feature.numpy(), preinputs.test_sim_att_1.numpy(), preinputs.test_sim_att_2.numpy(), preinputs.test_sim_label.numpy()
    cur = 0
    G_unseen_sample = np.zeros([0, 2048])
    while(True):
        if cur+opt.batchsize >= len(test_sim_feature):
            temp_x = test_sim_feature[cur:cur+opt.batchsize]
            temp_att = test_sim_att_2[cur:cur+opt.batchsize]
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            g_sample, _ = netG(temp_x, temp_att)
            G_unseen_sample = np.vstack((G_unseen_sample, g_sample.numpy()))
            cur = cur + opt.batchsize
        else:
            temp_x = test_sim_feature[cur:]
            temp_att = test_sim_att_2[cur:]
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            g_sample, _ = netG(temp_x, temp_att)
            G_unseen_sample = np.vstack((G_unseen_sample, g_sample.detach().cpu().numpy()))
            break

    # test_seen
    test_seen_sim_feature, test_seen_sim_att_2, test_seen_sim_label = preinputs.test_seen_sim_feature.numpy(), preinputs.test_seen_sim_att_2.numpy(), preinputs.test_seen_sim_label.numpy()
    cur = 0
    G_seen_sample = np.zeros([0, 2048])
    while(True):
        if cur+opt.batchsize >= len(test_seen_sim_feature):
            temp_x = test_seen_sim_feature[cur:cur+opt.batchsize]
            temp_att = test_seen_sim_att_2[cur:cur+opt.batchsize]
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            g_sample, _ = netG(temp_x, temp_att)
            G_seen_sample = np.vstack((G_seen_sample, g_sample.numpy()))
            cur = cur + opt.batchsize
        else:
            temp_x = test_seen_sim_feature[cur:]
            temp_att = test_seen_sim_att_2[cur:]
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            g_sample, _ = netG(temp_x, temp_att)
            G_seen_sample = np.vstack((G_seen_sample, g_sample.detach().cpu().numpy()))
            break

    G_sample = np.vstack((G_seen_sample, G_unseen_sample))

    slabel = test_seen_sim_label.reshape(-1,1)
    ulabel = test_sim_label.reshape(-1,1)
    G_label = np.vstack((slabel, ulabel))

    unseen_label = []
    for i in dataset.test_unseen_label.numpy():
        unseen_label.append(i+dataset.ntrain_class)
    unseen_label = np.array(unseen_label)

    classifier = Softmax(G_sample, G_label)
    
    # S-->T
    acc_S_T = classifier.acc(dataset.test_seen_feature.numpy(), dataset.test_seen_label.numpy())

    # U-->T
    acc_U_T = classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)

    acc = (2 * acc_S_T * acc_U_T) / (acc_S_T + acc_U_T)

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.best_acc_S_T = acc_S_T
        result.best_acc_U_T = acc_U_T
        result.save_model = True

    print("H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(acc, acc_S_T, acc_U_T))


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c

def compute_dissimilarity(data, center):
    dissimilarity = Variable(torch.empty([data.size(0), center.size(0)]))
    cos = nn.CosineSimilarity(dim=1)
    for i,d in enumerate(data):
        d = d.view((1,-1))
        dissimilarity[i] = cos(d, center)   
    return dissimilarity.cuda()


if __name__ == "__main__":
    train()

