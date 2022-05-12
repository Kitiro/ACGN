# coding=utf8
# 训练基于连续属性的属性分类器
from sklearn.externals import joblib
from sklearn.svm import LinearSVC, SVC
from dataset import DATA_LOADER
import argparse
import numpy as np
import os, sys
from sklearn import preprocessing

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


if __name__=='__main__':
    opt = parser.parse_args()
    scaler = preprocessing.MinMaxScaler()
    dataset = DATA_LOADER(opt)

    att_dim = dataset.attribute.numpy().shape[1]
    feat_dim = 2048
    trainX = dataset.train_feature.numpy()
    labels = dataset.train_label.numpy()
    attribute = dataset.attribute.numpy()
    scalered_att = scaler.fit_transform(attribute)
    train_att = scalered_att[dataset.seenclasses.numpy()]

    save_path = '/data/liujinlu/zsl/Cross_Class_GAN/data/{}/att_classifiers'.format(opt.dataset)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(att_dim):
        trainY = []   
        for j in labels:
            value = train_att[j][i]
            l = int(round((value * 10)))
            trainY.append(l)

        trainY=np.array(trainY)
        # print(trainX.shape, trainY.shape)
        if len(np.unique(trainY))>1:
            model = SVC(probability=True).fit(trainX,trainY)
            # model = LinearSVC().fit(trainX,trainY)
            joblib.dump(model, save_path+'/{}.m'.format(i))
            joblib.dump(np.unique(trainY), save_path+'/{}_labels.m'.format(i))
            print(np.unique(trainY))
            print(i, '-------------------finish----------------')
        else:
            pass



