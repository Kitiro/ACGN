# coding=utf8
# 基于属性分类器筛选样例
# 实验证明筛选无效，用筛选后的样例再进行训练分类，结果没有提高
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from dataset import DATA_LOADER, KnnFeat, SVM, Softmax
from models import _netCC_3, _param
import argparse
import numpy as np
import os, sys
from sklearn import preprocessing

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2')
parser.add_argument('--dataroot', default='/data/liujinlu/xian_resnet101/xlsa17/data/',
                    help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_false', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--split', type=str, default='Nongan')
parser.add_argument('--gpu', default='3', type=str, help='index of GPU to use')

parser.add_argument('--random', action='store_true', default=False, help = 'random pairs in data preparation')
parser.add_argument('--batchsize', type = int, default = 1024, help= 'batchsize')
parser.add_argument('--k_class', type = int, default = 1, help= 'find k similar classes')
parser.add_argument('--k_inst', type = int, default = 1, help= 'find k similar instances in each similar class')
parser.add_argument('--threshold', type = str, default = 'mean', help= 'use mean or dedian value as threshold')
parser.add_argument('--datadir', type = str, default = '', help= 'model datadir')


def test_zsl(datadir, test_input, netG, opt, dataset):
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

    # classifier = SVM(G_sample, test_sim_label)
    classifier = Softmax(G_sample, test_sim_label)
    acc = classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)
    
    new_acc = 0
    count = 0
    while new_acc <= acc and count < 3:
        count += 1
        screened_sample, screened_label = sample_screen_unseen(G_sample, test_sim_label, opt)
        new_classifier = Softmax(screened_sample, screened_label)
        new_acc = new_classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)
        G_sample = screened_sample
        test_sim_label = screened_label

        print("Accuracy: {:.2f}%  New Accuracy: {:.2f}%".format(acc, new_acc))

    acclog = "Ori Acc: {:.2f}%  New Acc: {:.2f}% -- {}".format(acc, new_acc, opt.threshold)+'\n'
    with open(datadir+'/screened_zsl_acc_{:.2f}_{}.txt'.format(new_acc, opt.threshold), 'w') as file:
        file.write(acclog)

def test_gzsl(datadir, test_input, netG, opt, dataset):
    # test_unseen
    test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label = test_input.test_sim_feature.numpy(), test_input.test_sim_att_1.numpy(), test_input.test_sim_att_2.numpy(), test_input.test_sim_label.numpy()
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
    test_seen_sim_feature, test_seen_sim_att_2, test_seen_sim_label = test_input.test_seen_sim_feature.numpy(), test_input.test_seen_sim_att_2.numpy(), test_input.test_seen_sim_label.numpy()
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
    slabel = test_seen_sim_label.reshape(-1,1)
    ulabel = test_sim_label.reshape(-1,1)
    G_label = np.vstack((slabel, ulabel))

    unseen_label = []
    for i in dataset.test_unseen_label.numpy():
        unseen_label.append(i+dataset.ntrain_class)
    unseen_label = np.array(unseen_label)


    G_sample = np.vstack((G_seen_sample, G_unseen_sample))
    classifier = Softmax(G_sample, G_label)

    ###  before sample screen   
    # S-->T
    acc_S_T = classifier.acc(dataset.test_seen_feature.numpy(), dataset.test_seen_label.numpy())
    # U-->T
    acc_U_T = classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)
    # H
    acc = (2 * acc_S_T * acc_U_T) / (acc_S_T + acc_U_T)

    ### after sample screen
    screened_sample, screened_label = sample_screen_all(G_sample, G_label, opt)
    new_classifier = Softmax(screened_sample, screened_label)
    # S-->T
    new_acc_S_T = new_classifier.acc(dataset.test_seen_feature.numpy(), dataset.test_seen_label.numpy())
    # U-->T
    new_acc_U_T = new_classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)
    # H
    new_acc = (2 * new_acc_S_T * new_acc_U_T) / (new_acc_S_T + new_acc_U_T)


    print("Ori H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(acc, acc_S_T, acc_U_T))
    print("New H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(new_acc, new_acc_S_T, new_acc_U_T))

    acclog = "Ori H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(acc, acc_S_T, acc_U_T)+'\n' + \
        "New H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(new_acc, new_acc_S_T, new_acc_U_T) + '\n'
    with open(datadir+'/screened_gzsl_H_{:.2f}_S_{:.2f}_T_{:.2f}_{}.txt'.format(new_acc, new_acc_S_T, new_acc_U_T, opt.threshold), 'w') as file:
        file.write(acclog)


def sample_screen_unseen(test_data, test_label, opt):
    models = []
    none_list = []
    labels_list = [] # record labels for each model
    for i in range(opt.att_dim):
        if os.path.exists(opt.classfier_root+'/{}.m'.format(i)):
            model = joblib.load(opt.classfier_root+'/{}.m'.format(i))
            models.append(model)
            l = joblib.load(opt.classfier_root+'/{}_labels.m'.format(i))
            labels_list.append(l.tolist())
        else:
            models.append('None')
            labels_list.append('None')
            none_list.append(i)
    scores = []
    for data, claind in zip(test_data,test_label):
        total_score = 1
        for i in range(opt.att_dim): 
            if i not in none_list:
                att_value = int(np.round(opt.test_att[claind-opt.ntrain, i]*10))
                # score = models[i].score(data.reshape(1,-1), np.array([att_value]))
                pred = models[i].predict_proba(data.reshape(1,-1))
                if att_value in labels_list[i]:
                    score = pred[0][labels_list[i].index(att_value)]
                else:
                    att_up = att_value + 1
                    att_down = att_value - 1
                    try:
                        score = np.max((pred[0][labels_list[i].index(att_up)], pred[0][labels_list[i].index(att_down)]))
                    except:
                        score = 0
                if score:
                    total_score *= score
        # print(total_score)
        scores.append(total_score)
    scores = np.array(scores)
    transfered = np.empty([0,2048])
    labels = []
    for i in np.unique(test_label):
        temp_data = test_data[np.where(test_label==i)[0]] 

        temp_scores = scores[np.where(test_label==i)[0]]
        if opt.threshold=='mean':
            value = np.mean(temp_scores)
        if opt.threshold=='median':
            value = np.median(temp_scores)
        print(value)
        screened_sample = temp_data[np.where(temp_scores>=value)[0]]
        ori_cent = np.mean(temp_data, axis=0)
        new_cent = np.mean(screened_sample, axis=0)
        dis = new_cent - ori_cent
        trans_sample = temp_data - dis

        transfered = np.vstack((transfered, trans_sample))
        labels.extend(i for j in temp_data)

    # screened_label = test_label[np.where(scores>value)[0]]
    # return(screened_sample, screened_label)
    return(transfered, np.array(labels))



def sample_screen_all(test_data, test_label, opt):
    models = []
    none_list = []
    labels_list = [] # record labels for each model
    for i in range(opt.att_dim):
        if os.path.exists(opt.classfier_root+'/{}.m'.format(i)):
            model = joblib.load(opt.classfier_root+'/{}.m'.format(i))
            models.append(model)
            l = joblib.load(opt.classfier_root+'/{}_labels.m'.format(i))
            labels_list.append(l.tolist())
        else:
            models.append('None')
            labels_list.append('None')
            none_list.append(i)
    scores = []
    for data, claind in zip(test_data,test_label):
        total_score = 1
        for i in range(opt.att_dim):
            if i not in none_list:
                att_value = int(np.round(opt.attribute[claind, i]*10))
                # score = models[i].score(data.reshape(1,-1), np.array([att_value]))
                pred = models[i].predict_proba(data.reshape(1,-1))
                if att_value in labels_list[i]:
                    score = pred[0][labels_list[i].index(att_value)]
                else:
                    att_up = att_value + 1
                    att_down = att_value - 1
                    try:
                        score = np.max((pred[0][labels_list[i].index(att_up)], pred[0][labels_list[i].index(att_down)]))
                    except:
                        score = 0
                if score:
                    total_score *= score
        scores.append(total_score)
    scores = np.array(scores)
    transfered = np.empty([0,2048])
    labels = []
    for i in np.unique(test_label):
        temp_data = test_data[np.where(test_label==i)[0]] 
        temp_scores = scores[np.where(test_label==i)[0]]
        if opt.threshold=='mean':
            value = np.mean(temp_scores)
        if opt.threshold=='median':
            value = np.median(temp_scores)
        screened_sample = temp_data[np.where(temp_scores>=value)[0]]
        ori_cent = np.mean(temp_data, axis=0)
        new_cent = np.mean(screened_sample, axis=0)
        dis = new_cent - ori_cent
        trans_sample = temp_data + dis
        transfered = np.vstack((transfered, trans_sample))
        labels.extend(i for j in temp_data)

    # screened_label = test_label[np.where(scores>value)[0]]
    # return(screened_sample, screened_label)
    return(transfered, np.array(labels))



if __name__=='__main__':
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    scaler = preprocessing.MinMaxScaler()
    dataset = DATA_LOADER(opt)
    
    attribute = dataset.attribute.numpy()
    scalered_att = scaler.fit_transform(attribute)
    opt.test_att = scalered_att[dataset.unseenclasses.numpy()]
    opt.train_att = scalered_att[dataset.seenclasses.numpy()]
    opt.attribute = scalered_att
    # print(opt.attribute.shape)
    opt.att_dim = opt.test_att.shape[1]
    opt.ntrain = dataset.ntrain_class

    if opt.gpu!='-1':
        netG = _netCC_3(dataset.train_att.numpy().shape[1]).cuda()
    else:
        netG = _netCC_3(dataset.train_att.numpy().shape[1])

    opt.path_root = '/data/liujinlu/zsl/Cross_Class_GAN/data/{}/{}'.format(opt.dataset, opt.split)
    opt.classfier_root = '/data/liujinlu/zsl/Cross_Class_GAN/data/{}/att_classifiers'.format(opt.dataset)
    
    if opt.datadir != '':
        keys = opt.datadir.split('/')[-1]
        keys = keys.split('_')
        if keys[0]=='Random':
                opt.random = True
        else:
            opt.k_class = int(keys[0][-1])
            opt.k_inst = int(keys[1][-1])
        for model in os.listdir(opt.datadir):
            if model[:14]=='Best_model_ZSL':
                test_input = KnnFeat(opt)
                zsl_model = torch.load(os.path.join(opt.datadir, model))
                netG.load_state_dict(zsl_model['state_dict_G'])
                netG.eval()
                test_zsl(opt.datadir, test_input, netG, opt, dataset)

            if model[:15]=='Best_model_GZSL':
                test_input = KnnFeat(opt)
                gzsl_model = torch.load(os.path.join(opt.datadir, model))
                netG.load_state_dict(gzsl_model['state_dict_G'])
                netG.eval()
                test_gzsl(opt.datadir, test_input, netG, opt, dataset)

    else:           
        for file in os.listdir(opt.path_root):
            datadir = os.path.join(opt.path_root, file)
            keys = file.split('_')
            if keys[0]=='Random':
                    opt.random = True
            else:
                opt.k_class = int(keys[0][-1])
                opt.k_inst = int(keys[1][-1])

            for model in os.listdir(os.path.join(opt.path_root, file)):            
                
                if model[:14]=='Best_model_ZSL':
                    test_input = KnnFeat(opt)
                    zsl_model = torch.load(os.path.join(opt.path_root, file, model))
                    netG.load_state_dict(zsl_model['state_dict_G'])
                    netG.eval()
                    test_zsl(datadir, test_input, netG, opt, dataset)

                # if model[:15]=='Best_model_GZSL':
                #     test_input = KnnFeat(opt)
                #     gzsl_model = torch.load(os.path.join(opt.path_root, file, model))
                #     netG.load_state_dict(gzsl_model['state_dict_G'])
                #     netG.eval()
                #     test_gzsl(datadir, test_input, netG, opt, dataset)



