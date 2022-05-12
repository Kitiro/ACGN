# coding:utf8
# intra-class entropy
# 计算已知类的类内熵 E，相对类内熵 RE

import numpy as np
import argparse
import os
import scipy.io as sio
import h5py
from sklearn.neighbors import NearestNeighbors

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

from dataset import DATA_LOADER

opt = parser.parse_args()

dataset = DATA_LOADER(opt)

tr_feature = dataset.train_feature.numpy()
tr_label = dataset.train_label.numpy()
s_class = dataset.seenclasses.numpy()

matfile = sio.loadmat(os.path.join(opt.dataroot,opt.dataset,'att_splits.mat'))
allclasses = []
for c in matfile['allclasses_names']:
    allclasses.append(c[0][0])


E = [] # 类内熵 intra-class entropy
dic={} # 用于记录每个已知类的样本的最相似类标签，标签的排序只有已知类，比如AWA数据集的：0-39
for c_ind in range(len(s_class)):
    print('-------- '+allclasses[s_class[c_ind]]+' --------')
    dic[allclasses[s_class[c_ind]]] = []
    temp_data = tr_feature[np.where(tr_label!=c_ind)[0]]
    temp_label = tr_label[np.where(tr_label!=c_ind)[0]]
    knn = NearestNeighbors(n_neighbors = 1, metric = 'cosine').fit(temp_data)
    for sample in tr_feature[np.where(tr_label==c_ind)[0]]:
        d, s_ind = knn.kneighbors(sample.reshape((1,-1)))
        dic[allclasses[s_class[c_ind]]].append(temp_label[s_ind[0]])
        # print(s_class[temp_label[s_ind[0]]][0])
    
    sim_labels = np.array(dic[allclasses[s_class[c_ind]]])
    sim_cla = np.unique(sim_labels) # 该类中所有样本的最相似样本所在的类的集合
    temp_E = 0
    for i in sim_cla:
        p_i = len(sim_labels[np.where(sim_labels==i)[0]])/len(sim_labels)
        temp_E += (-p_i*np.log2(p_i))
    E.append(temp_E)
    print(temp_E)

E = np.array(E)
min_E = min(E)

RE = [] # 相对类内熵 relative E
for e in E:
    RE.append(e/min_E)
print(RE)

savepath = '/data/liujinlu/zsl/Cross_Class_GAN/data/' + opt.dataset
file = sio.savemat(savepath+'/entropy.mat',{'allclasses':allclasses, 'seenclasses':s_class,\
    'label_dic':dic, 'entropy':E, 'relative_entropy': RE})




