#coding:utf8
# 将生成的样本存储，为tsne图准备

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn


import numpy as np
import argparse
import os
import random
import json
import scipy.io as sio
import h5py
from dataset import FeatDataLayer, DATA_LOADER, KnnFeat
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
parser.add_argument('--evl_interval',  type=int, default=20)

parser.add_argument('--random', action='store_true', default=False, help = 'random pairs in data preparation')
parser.add_argument('--endEpoch', type = int, default = 5000, help= 'train epoch')
parser.add_argument('--gzsl', action='store_true', default = False, help = 'gzsl evaluation')
parser.add_argument('--batchsize', type = int, default = 1024, help= 'batchsize')
parser.add_argument('--k_class', type = int, default = 1, help= 'find k similar classes')
parser.add_argument('--k_inst', type = int, default = 1, help= 'find k similar instances in each similar class')
parser.add_argument('--att_w', type = int, default = 10, help= 'weight of Att_loss')
parser.add_argument('--x_w', type = int, default = 40, help= 'weight of X_loss')

opt = parser.parse_args()
opt.path_root = '/data/liujinlu/zsl/Cross_Class_GAN/data/%s' %opt.dataset





def eval_and_save(netG, preinputs, opt, dataset):
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
    if not opt.random:
        file = sio.savemat(opt.path_root+'/generated_data_cla{}_inst{}.mat'.format(opt.k_class, opt.k_inst),\
                {'data':G_sample, 'labels':test_sim_label})
    else:
        file = sio.savemat(opt.path_root+'/generated_data_random.mat',\
                {'data':G_sample, 'labels':test_sim_label})

    


def readdata():

    dataset = DATA_LOADER(opt)
    preinputs = KnnFeat(opt)
    if opt.random:
        data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.numpy(), preinputs.train_sim_feature.numpy(), preinputs.train_sim_att_1.numpy(), preinputs.train_sim_att_2.numpy(), opt)
    else:
        data_layer = FeatDataLayer(preinputs.train_sim_label.numpy(), preinputs.train_sim_output.numpy(), preinputs.train_sim_feature.numpy(), preinputs.train_sim_att_1.numpy(), preinputs.train_sim_att_2.numpy(), opt)
    

    netG = _netCC_3(preinputs.train_sim_att_2.numpy().shape[1]).cuda()


    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    netG.eval()
    eval_and_save(netG, preinputs, opt, dataset)


if __name__=='__main__':
    readdata()



