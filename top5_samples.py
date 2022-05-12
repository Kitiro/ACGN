# coding=utf8
# experimens: top-5 similar samples
# 用于nearest neigbor matching，以awa2数据集未见类为例

from dataset import DATA_LOADER, KnnFeat, SVM, Softmax
from models import _netCC_3, _param
import argparse
import numpy as np
import os, sys
import h5py
import scipy.io as sio
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AwA2')
parser.add_argument('--dataroot', default='data',
                    help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_false', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--gpu', default='1', type=str, help='index of GPU to use')

parser.add_argument('--random', action='store_true', default=False, help = 'random pairs in data preparation')
parser.add_argument('--batchsize', type = int, default = 1024, help= 'batchsize')
parser.add_argument('--k_class', type = int, default = 1, help= 'find k similar classes')
parser.add_argument('--k_inst', type = int, default = 1, help= 'find k similar instances in each similar class')
parser.add_argument('--modeldir', type = str, default = '', help= 'modeldir, only for ZSL model')
parser.add_argument('--savedir', type = str, default = '', help= 'savedir, only for ZSL model')

def test_zsl(test_input, netG, opt, dataset):
    test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label = test_input.test_sim_feature.numpy(), test_input.test_sim_att_1.numpy(), test_input.test_sim_att_2.numpy(), test_input.test_sim_label.numpy()
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

    return(G_sample, test_sim_label, unseen_label)
    
    


if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.modeldir=='' or opt.savedir=='':
        print('dirs can not be empty.')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
        dataset = DATA_LOADER(opt)

        keys = opt.modeldir.split('/')[-2]
        keys = keys.split('_')
        if keys[0]=='Random':
                opt.random = True
        else:
            opt.k_class = int(keys[0][-1])
            opt.k_inst = int(keys[1][-1])
        test_input = KnnFeat(opt)

        if opt.gpu!='-1':
            netG = _netCC_3(dataset.train_att.numpy().shape[1]).cuda()
        else:
            netG = _netCC_3(dataset.train_att.numpy().shape[1])

        zsl_model = torch.load(opt.modeldir)
        netG.load_state_dict(zsl_model['state_dict_G'])
        netG.eval()
        G_sample, G_label, R_label = test_zsl(test_input, netG, opt, dataset)
        R_sample = dataset.test_unseen_feature.numpy()
        knn_index = NearestNeighbors(n_neighbors = 5, metric = 'cosine').fit(R_sample)
        knn_prob  = KNeighborsClassifier(n_neighbors = 5, metric = 'cosine').fit(R_sample, R_label)

        ## TO DO 
        imgs = dataset.image_files
        unseen_imgs = imgs[dataset.test_unseen_loc]
        unseen_class = dataset.unseenclasses.numpy()

        tt_label = np.unique(G_label)
        # rand_cla = np.random.randint(tt_label[0],tt_label[-1], 3)
        clanames = []
        imgnames = []
        # for cla in tt_label:
        #     print(cla)
        #     imgname = []
        #     probs = []
        #     claind = unseen_class[cla-dataset.ntrain_class]
        #     clanames.append(claind)
        #     temp_data = G_sample[np.where(G_label==cla)[0]] #所有属于该类的生成sample
        #     for sample in temp_data:
        #         d, ind = knn_index.kneighbors(sample.reshape(1,-1)) #生成sample的top5特征相似图片 from all samples of unseen classes
        #         probs.append(pred_proba)
        #         imgname.append(unseen_imgs[ind[0]]) 
        #     imgnames.append(imgname)
        # sio.savemat(opt.savedir+'/top5files.mat',{'classes':clanames, 'imgs':imgnames})
        all_probs = []
        for cla in tt_label:
            print(cla)
            probs = []
            claind = unseen_class[cla-dataset.ntrain_class]
            clanames.append(claind)
            temp_data = G_sample[np.where(G_label==cla)[0]] #所有属于该类的生成sample
            for sample in temp_data:
                pred_proba = knn_prob.predict_proba(sample.reshape(1,-1))[0] # 利用KNN预测该生成sample的label
                probs.append(pred_proba)
            all_probs.append(probs)
        sio.savemat(opt.savedir+'/5nn_prob.mat',{'classes':clanames, 'probs':np.array(all_probs)})

        


