# coding:utf8
from torch import nn
import numpy as np
import scipy.io as sio
from termcolor import cprint
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import torch
import h5py
import os
from sklearn.svm import LinearSVC
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as init
from sklearn.cluster import KMeans


def map_label(label, classes):
    mapped_label = torch.zeros(label.size(0), dtype=torch.long)
    for idx, cla in enumerate(classes):
        mapped_label[label == int(cla)] = idx
    return mapped_label


# source : [1*2048], target : [batch, 2048]
def find_similar_samples(source, target, k):
    diff = np.tile(source, (target.shape[0], 1)) - target
    distance = np.sum(diff ** 2, axis=1) ** 0.5
    max_index = np.argsort(distance)[:k]
    return target[max_index]


SYN_NUM = 500  # Synthesized samples number
# 获取数据集
class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == "imageNet1K":
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]
        self.tr_cls_centroid = np.zeros(
            [self.train_cls_num+self.test_cls_num, self.feature_dim], np.float32
        )  # .astype(np.float32)
        for i in self.seenclasses:
            self.tr_cls_centroid[i] = np.mean(
                self.train_feature[self.train_label == i].numpy(), axis=0
            )

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print("MinMaxScaler...")
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(
                opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat",
                "r",
            )
            feature = scaler.fit_transform(np.array(matcontent["features"]))
            label = np.array(matcontent["labels"]).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent["features_val"]))
            label_val = np.array(matcontent["labels_val"]).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File(
                "/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat",
                "r",
            )
            feature_unseen = scaler.transform(np.array(matcontent["features"]))
            label_unseen = np.array(matcontent["labels"]).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(
                opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat",
                "r",
            )
            feature = np.array(matcontent["features"])
            label = np.array(matcontent["labels"]).astype(int).squeeze() - 1
            feature_val = np.array(matcontent["features_val"])
            label_val = np.array(matcontent["labels_val"]).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(
            opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat"
        )
        self.attribute = torch.from_numpy(matcontent["w2v"]).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    # self.attribute(50,85)  self.allclasses(50)     self.seenclasses(40)        self.test_att(10,85)    self.test_seen_feature(5882,2048)
    # test_seen_label (5882)     self.test_seen_loc(5882)    test_unseen_feature(7913,2048)      test_unseen_label(7913)
    # test_unseen_loc(7913)      train_att(40,85)        train_class(40)     train_feature(23527,2048)   train_label(23527)      trainval_loc(23527)
    # unseenclasses(10)  val_unseen_loc(9191)
    def read_matdataset(self, opt):

        matcontent = sio.loadmat(
            opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat"
        )
        self.image_files = matcontent["image_files"]
        feature = matcontent["features"].T
        label = matcontent["labels"].astype(int).squeeze() - 1
        matcontent = sio.loadmat(
            opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat"
        )
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent["trainval_loc"].squeeze() - 1
        train_loc = matcontent["train_loc"].squeeze() - 1
        val_unseen_loc = matcontent["val_loc"].squeeze() - 1
        test_seen_loc = matcontent["test_seen_loc"].squeeze() - 1
        test_unseen_loc = matcontent["test_unseen_loc"].squeeze() - 1

        self.trainval_loc = trainval_loc
        self.train_loc = train_loc
        self.val_unseen_loc = val_unseen_loc
        self.test_seen_loc = test_seen_loc
        self.test_unseen_loc = test_unseen_loc

        # self.attribute = torch.from_numpy(matcontent["att"].T).float()

        self.attribute = preprocessing.scale(matcontent["att"].T, axis=0)
        self.attribute = torch.from_numpy(self.attribute).float()

        # if opt.extended_attr_num != 0:
        #     attr_path = "/home/zzc/exp/Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/generate_attributes/generated_attributes"
        #     attribute2 = np.load(
        #         os.path.join(attr_path, "class_attribute_map_"+opt.dataset+".npy")
        #     )
        #     attribute2 = attribute2[:, : opt.extended_attr_num]

        #     self.attribute = np.hstack((self.attribute, attribute2))
        #     self.attribute = preprocessing.scale(self.attribute, axis=1)

        # glovefile = h5py.File('/data/liujinlu/zsl_glove/data/%s/glove_features.h5'%opt.dataset, 'r')
        # text_feat = glovefile['glove_features'][:] # (n_instances, 300d)
        # text = glovefile['classes_vectors'][:] # (n_classes, 300d)
        # glovefile.close()
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print("standardization...")
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

                self.train_feature = torch.from_numpy(_train_feature).float()

                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(
                    _test_unseen_feature
                ).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

                # _text_feat = scaler.fit_transform(text_feat)
                # _text = scaler.fit_transform(text)
                # self.train_text_feat = torch.from_numpy(_text_feat[trainval_loc]).float()
                # self.test_unseen_text_feat = torch.from_numpy(_text_feat[test_unseen_loc]).float()
                # self.test_seen_text_feat = torch.from_numpy(_text_feat[test_seen_loc]).float()
                # self.class_text = torch.from_numpy(_text).float()

            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(
                    feature[test_unseen_loc]
                ).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(
                    feature[test_seen_loc]
                ).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

                # self.train_text_feat = torch.from_numpy(text_feat[trainval_loc]).float()
                # self.test_unseen_text_feat = torch.from_numpy(text_feat[test_unseen_loc]).float()
                # self.test_seen_text_feat = torch.from_numpy(text_feat[test_seen_loc]).float()
                # self.class_text = torch.from_numpy(text).float()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

            # self.train_text_feat = torch.from_numpy(text_feat[trainval_loc]).float()
            # self.test_unseen_text_feat = torch.from_numpy(text_feat[test_unseen_loc]).float()
            # self.test_seen_text_feat = torch.from_numpy(text_feat[test_seen_loc]).float()
            # self.class_text = torch.from_numpy(text).float()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        # self.train_label = map_label(self.train_label, self.seenclasses)
        # self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        # self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses]
        self.test_att = self.attribute[self.unseenclasses]
        print("123.....................")
        # self.train_cls_num = 150
        # self.test_cls_num  = 50

        # self.train_text = self.class_text[self.seenclasses].numpy()
        # self.test_text = self.class_text[self.unseenclasses].numpy()


class FeatDataLayer(object):
    def __init__(self, label, feat_data, x, att2, opt):
        """Set the roidb to be used by this layer during training."""
        # self._roidb = roidb
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data  # output feature
        self._label = label
        self._x = x  # input feature
        self._att2 = att2
        self._shuffle_roidb_inds()
        self._epoch = 0

        cprint(
            "Whole data length:{}, need {} iterations.".format(
                len(self._x), int(len(self._x) / self._opt.batchsize) + 1
            ),
            "blue",
        )

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur : self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        new_epoch = False
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur : self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_x = np.array([self._x[i] for i in db_inds])
        minibatch_att2 = np.array([self._att2[i] for i in db_inds]).squeeze()
        minibatch_label = np.array([self._label[i] for i in db_inds])

        blobs = {
            "data": minibatch_feat,
            "labels": minibatch_label,
            "x": minibatch_x,
            "att2": minibatch_att2,
            "newEpoch": new_epoch,
        }
        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {"data": self._feat_data, "labels": self._label}
        return blobs


class TestKnn(object):
    """ prepare data for zsl, gzsl. at test stage
        to test trained-over models
    """

    def __init__(self, opt):
        self.dataset = DATA_LOADER(opt)
        self.datapath = "/data/liujinlu/zsl/Cross_Class_GAN/data"
        self.kcla = 4
        self.kinst = [1, 5, 10]
        (
            self.seen_source_input,
            self.seen_target_label,
            self.seen_target_att,
        ) = self._get_test_seen_data()
        (
            self.unseen_source_input,
            self.unseen_target_label,
            self.unseen_target_att,
        ) = self._get_test_unseen_data()

    def _get_test_seen_data(self):
        #
        trclass_dic = {}  # 记录每一个已知类在train_classes中的最相似类的序号
        trclass_knn = NearestNeighbors(n_neighbors=self.kcla + 1, metric="cosine").fit(
            self.dataset.train_att
        )
        for i in np.unique(self.dataset.train_label.numpy()):
            d, ind = trclass_knn.kneighbors(self.dataset.train_att[i].reshape((1, -1)))
            trclass_dic[i] = ind[0][1:]
        # prepare source input for test_seen data
        seen_source_input = {}
        seen_target_att = {}
        seen_target_label = {}
        for kc in range(self.kcla):
            for ki in self.kinst:
                seen_source_input["c" + str(kc) + "_" + str(ki)] = []
                seen_target_att["c" + str(kc) + "_" + str(ki)] = []
                seen_target_label["c" + str(kc) + "_" + str(ki)] = []
        for kc in range(self.kcla):
            for ki in self.kinst:
                seen_source_input["r" + str(kc) + "_" + str(ki)] = []
                seen_target_att["r" + str(kc) + "_" + str(ki)] = []
                seen_target_label["r" + str(kc) + "_" + str(ki)] = []

        for kc in range(self.kcla):
            for ki in self.kinst:
                for i in np.unique(self.dataset.test_seen_label.numpy()):
                    for cla in trclass_dic[i][: kc + 1]:
                        temp_feat = self.dataset.train_feature.numpy()[
                            np.where(self.dataset.train_label.numpy() == cla)[0]
                        ]
                        temp_center = np.mean(temp_feat, axis=0)
                        temp_knn = NearestNeighbors(
                            n_neighbors=ki + 1, metric="euclidean"
                        ).fit(temp_feat)
                        # print(temp_feat.shape)
                        # print(temp_center.shape)
                        d, ind = temp_knn.kneighbors(temp_center.reshape((1, -1)))
                        seen_source_input["c" + str(kc) + "_" + str(ki)].extend(
                            temp_feat[inx] for inx in ind[0]
                        )
                        seen_target_label["c" + str(kc) + "_" + str(ki)].extend(
                            i for inx in ind[0]
                        )
                        seen_target_att["c" + str(kc) + "_" + str(ki)].extend(
                            self.dataset.train_att[i].numpy() for inx in ind[0]
                        )

                        ranind = np.random.randint(0, len(temp_feat), ki)
                        seen_source_input["r" + str(kc) + "_" + str(ki)].extend(
                            temp_feat[inx] for inx in ranind
                        )
                        seen_target_label["r" + str(kc) + "_" + str(ki)].extend(
                            i for inx in ranind
                        )
                        seen_target_att["r" + str(kc) + "_" + str(ki)].extend(
                            self.dataset.train_att[i].numpy() for inx in ranind
                        )

        for k in seen_source_input.keys():

            seen_source_input[k] = np.array(seen_source_input[k])
            seen_source_input[k] = torch.from_numpy(seen_source_input[k]).float()
            seen_target_label[k] = np.array(seen_target_label[k])
            seen_target_label[k] = torch.from_numpy(seen_target_label[k]).long()
            seen_target_att[k] = np.array(seen_target_att[k])
            seen_target_att[k] = torch.from_numpy(seen_target_att[k]).float()

        return seen_source_input, seen_target_label, seen_target_att

    def _get_test_unseen_data(self):

        ttclass_dic = {}  # 记录每一个未知类在train_classes中的最相似类的序号
        ttclass_knn = NearestNeighbors(n_neighbors=self.kcla, metric="cosine").fit(
            self.dataset.train_att
        )
        for i in np.unique(self.dataset.test_unseen_label.numpy()):
            d, ind = ttclass_knn.kneighbors(self.dataset.test_att[i].reshape((1, -1)))
            ttclass_dic[i] = ind[0][:]

        # prepare source input for test_unseen data
        unseen_source_input = {}
        unseen_target_att = {}
        unseen_target_label = {}
        for kc in range(self.kcla):
            for ki in self.kinst:
                unseen_source_input["c" + str(kc) + "_" + str(ki)] = []
                unseen_target_att["c" + str(kc) + "_" + str(ki)] = []
                unseen_target_label["c" + str(kc) + "_" + str(ki)] = []
        for kc in range(self.kcla):
            for ki in self.kinst:
                unseen_source_input["r" + str(kc) + "_" + str(ki)] = []
                unseen_target_att["r" + str(kc) + "_" + str(ki)] = []
                unseen_target_label["r" + str(kc) + "_" + str(ki)] = []

        for kc in range(self.kcla):
            for ki in self.kinst:
                for i in np.unique(self.dataset.test_unseen_label.numpy()):
                    for cla in ttclass_dic[i][: kc + 1]:
                        temp_feat = self.dataset.train_feature.numpy()[
                            np.where(self.dataset.train_label.numpy() == cla)[0]
                        ]
                        temp_center = np.mean(temp_feat, axis=0)
                        temp_knn = NearestNeighbors(
                            n_neighbors=ki + 1, metric="euclidean"
                        ).fit(temp_feat)

                        d, ind = temp_knn.kneighbors(temp_center.reshape((1, -1)))
                        unseen_source_input["c" + str(kc) + "_" + str(ki)].extend(
                            temp_feat[inx] for inx in ind[0]
                        )
                        unseen_target_label["c" + str(kc) + "_" + str(ki)].extend(
                            i + self.dataset.ntrain_class for inx in ind[0]
                        )
                        unseen_target_att["c" + str(kc) + "_" + str(ki)].extend(
                            self.dataset.test_att[i].numpy() for inx in ind[0]
                        )

                        ranind = np.random.randint(0, len(temp_feat), ki)
                        unseen_source_input["r" + str(kc) + "_" + str(ki)].extend(
                            temp_feat[inx] for inx in ranind
                        )
                        unseen_target_label["r" + str(kc) + "_" + str(ki)].extend(
                            i + self.dataset.ntrain_class for inx in ranind
                        )
                        unseen_target_att["r" + str(kc) + "_" + str(ki)].extend(
                            self.dataset.test_att[i].numpy() for inx in ranind
                        )
        for k in unseen_source_input.keys():
            unseen_source_input[k] = np.array(unseen_source_input[k])
            unseen_source_input[k] = torch.from_numpy(unseen_source_input[k]).float()
            unseen_target_label[k] = np.array(unseen_target_label[k])
            unseen_target_label[k] = torch.from_numpy(unseen_target_label[k]).long()
            unseen_target_att[k] = np.array(unseen_target_att[k])
            unseen_target_att[k] = torch.from_numpy(unseen_target_att[k]).float()
            # print(k)
            # print(unseen_source_input[k].size())
            # print(unseen_target_label[k].size())
            # print(unseen_target_att[k].size())

        return unseen_source_input, unseen_target_label, unseen_target_att


# 数据预处理
class KnnFeat_attention(object):
    """ prepare end-to-end training data, test_seen data, test_unseen data
    """

    def __init__(self, opt):
        self.dataset = DATA_LOADER(opt)

        self.datapath = "data"
        self.file_path = os.path.join(
            self.datapath,
            "%s/KnnFeat_%d_cla_%d_inst_attention.h5"
            % (opt.dataset, opt.k_class, opt.k_inst),
        )
        (
            self.train_sim_feature,
            self.train_sim_att_2,
            self.train_sim_output,
            self.train_sim_label,
            self.test_seen_sim_feature,
            self.test_seen_sim_att_2,
            self.test_seen_sim_label,
            self.test_sim_feature,
            self.test_sim_att_2,
            self.test_sim_label,
        ) = self._getKnnFeat(opt)

    def _getKnnFeat(self, opt):
        if opt.validation:
            # 暂时不用validation，没写
            return None
        else:
            if opt.random:
                """ 只从最相似类中随机找一个样本作为输入
                """
                # if os.path.isfile('/data/liujinlu/zsl_glove/data/%s/KnnFeat_random.h5'%opt.dataset):
                #     print('read data ...')
                #     file = h5py.File('/data/liujinlu/zsl_glove/data/%s/KnnFeat_random.h5'%opt.dataset, 'r')
                #     train_sim_feature = torch.from_numpy(file['train_sim_feature'][:]).float()
                #     train_sim_att_1 = torch.from_numpy(file['train_sim_att_1'][:].squeeze()).float()
                #     train_sim_att_2 = torch.from_numpy(file['train_sim_att_2'][:].squeeze()).float()
                #     test_sim_feature = torch.from_numpy(file['test_sim_feature'][:]).float()
                #     test_sim_att_1 = torch.from_numpy(file['test_sim_att_1'][:].squeeze()).float()
                #     test_sim_att_2 = torch.from_numpy(file['test_sim_att_2'][:].squeeze()).float()
                #     test_sim_label = torch.from_numpy(file['test_sim_label'][:]).long()
                #     test_seen_sim_feature = torch.from_numpy(file['test_seen_sim_feature'][:]).float()
                #     test_seen_sim_att_1 = torch.from_numpy(file['test_seen_sim_att_1'][:].squeeze()).float()
                #     test_seen_sim_att_2 = torch.from_numpy(file['test_seen_sim_att_2'][:].squeeze()).float()
                #     test_seen_sim_label = torch.from_numpy(file['test_seen_sim_label'][:]).long()
                #     file.close()
                #     print('read data done.')
                #     return train_sim_feature, train_sim_att_1,train_sim_att_2, \
                #     test_seen_sim_feature, test_seen_sim_att_1, test_seen_sim_att_2, test_seen_sim_label,\
                #             test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label
                # else:

                print("prepare data1 ...")
                cla_num = {}  # 记录每个训练类的样例个数
                for i in np.unique(self.dataset.train_label.numpy()):
                    cla_num[i] = len(
                        self.dataset.train_feature.numpy()[
                            np.where(self.dataset.train_label.numpy() == i)[0]
                        ]
                    )
                # 训练的输入
                class_dic = {}  # 通过semantic feature,找到每一个训练类在train_classes中的最相似类的序号
                class_knn = NearestNeighbors(
                    n_neighbors=opt.k_class + 1, metric="cosine"
                ).fit(self.dataset.train_att)
                for i in np.unique(self.dataset.train_label.numpy()):
                    d, ind = class_knn.kneighbors(
                        self.dataset.train_att[i].reshape((1, -1))
                    )
                    class_dic[i] = ind[0][1:]

                train_sim_feature = []
                train_sim_att_1 = []  # semantic feature of target class
                train_sim_att_2 = []  # semantic feature of source class
                train_sim_output = []  # no use
                train_sim_label = []  # no use
                for i, j in zip(
                    self.dataset.train_feature.numpy(), self.dataset.train_label.numpy()
                ):
                    seed = random.randint(
                        0, cla_num[class_dic[j][0]] - 1
                    )  # 从j类最相似的q类的样例中随机取一个, 样本一一对应。
                    train_sim_feature.append(
                        self.dataset.train_feature.numpy()[
                            np.where(self.dataset.train_label.numpy() == class_dic[j])[0]
                        ][seed]
                    )
                    # train_sim_att_1.append(self.dataset.train_att[class_dic[j]])
                    train_sim_att_2.append(self.dataset.train_att[j])
                train_sim_feature = np.array(train_sim_feature).squeeze()
                train_sim_feature = torch.from_numpy(
                    train_sim_feature
                ).float()  # label 和 train_label一致
                train_sim_att_1 = np.array(train_sim_att_1)
                train_sim_att_1 = torch.from_numpy(train_sim_att_1).float()
                train_sim_att_2 = np.array(train_sim_att_2)
                train_sim_att_2 = torch.from_numpy(train_sim_att_2).float()
                train_sim_output = np.array(train_sim_output).squeeze()
                train_sim_output = torch.from_numpy(train_sim_output).float()
                train_sim_label = np.array(train_sim_label).squeeze()
                train_sim_label = torch.from_numpy(train_sim_label).long()

                # 生成 test_seen_feature 时的输入
                test_seen_sim_feature = []
                test_seen_sim_att_1 = []
                test_seen_sim_att_2 = []
                test_seen_sim_label = []
                test_seen_sim_output = []  # no use
                for i in np.unique(self.dataset.test_seen_label.numpy()):
                    temp_feat = self.dataset.train_feature.numpy()[
                        np.where(self.dataset.train_label.numpy() == i)[0]
                    ]
                    test_seen_sim_feature.extend(temp_feat[0 : int(len(temp_feat))])
                    test_seen_sim_label.extend(i for j in range(int(len(temp_feat))))
                    test_seen_sim_att_1.extend(
                        self.dataset.train_att[i] for _ in range(int(len(temp_feat)))
                    )
                    test_seen_sim_att_2.extend(
                        self.dataset.train_att[i] for _ in range(int(len(temp_feat)))
                    )
                test_seen_sim_feature = np.array(test_seen_sim_feature).squeeze()
                test_seen_sim_feature = torch.from_numpy(test_seen_sim_feature).float()
                test_seen_sim_label = np.array(test_seen_sim_label)
                test_seen_sim_label = torch.from_numpy(test_seen_sim_label).long()
                test_seen_sim_att_2 = np.array(test_seen_sim_att_2).squeeze()
                test_seen_sim_att_2 = torch.from_numpy(test_seen_sim_att_2).float()
                test_seen_sim_att_1 = np.array(test_seen_sim_att_1).squeeze()
                test_seen_sim_att_1 = torch.from_numpy(test_seen_sim_att_1).float()
                test_seen_sim_output = np.array(test_seen_sim_output).squeeze()
                test_seen_sim_output = torch.from_numpy(test_seen_sim_output).float()

                # 生成 test_unseen_feature 时的输入
                test_sim_feature = []
                test_sim_att_1 = []
                test_sim_att_2 = []
                test_sim_label = []
                test_sim_output = []
                test_class_knn = NearestNeighbors(
                    n_neighbors=opt.k_class, metric="cosine"
                ).fit(self.dataset.train_att)

                for i in np.unique(self.dataset.test_unseen_label.numpy()):
                    # 用unseen的attribute去seen中找到与each unseen最相似的k个seen class
                    d, ind = test_class_knn.kneighbors(
                        self.dataset.test_att[i].reshape((1, -1))
                    )
                    temp_feat = self.dataset.train_feature.numpy()[
                        np.where(self.dataset.train_label.numpy() == ind[0])[0]
                    ]  # 与unsen class i最相似的seen class的visual feature
                    test_sim_feature.extend(temp_feat)
                    test_sim_label.extend(
                        i + self.dataset.ntrain_class for j in range(len(temp_feat))
                    )
                    test_sim_att_1.extend(
                        self.dataset.train_att[ind[0]] for j in range(len(temp_feat))
                    )  #  source
                    test_sim_att_2.extend(
                        self.dataset.test_att[i] for j in range(len(temp_feat))
                    )  # target
                test_sim_feature = np.array(test_sim_feature).squeeze()
                test_sim_feature = torch.from_numpy(test_sim_feature).float()
                test_sim_label = np.array(test_sim_label)
                test_sim_label = torch.from_numpy(test_sim_label).long()
                test_sim_att_1 = np.array(test_sim_att_1).squeeze()
                test_sim_att_1 = torch.from_numpy(test_sim_att_1).float()
                test_sim_att_2 = np.array(test_sim_att_2).squeeze()
                test_sim_att_2 = torch.from_numpy(test_sim_att_2).float()
                test_sim_output = np.array(test_sim_output).squeeze()
                test_sim_output = torch.from_numpy(test_sim_output).float()


                return (
                    train_sim_feature,
                    train_sim_att_1,
                    train_sim_att_2,
                    train_sim_output,
                    train_sim_label,
                    test_seen_sim_feature,
                    test_seen_sim_att_1,
                    test_seen_sim_att_2,
                    test_seen_sim_label,
                    test_seen_sim_output,
                    test_sim_feature,
                    test_sim_att_1,
                    test_sim_att_2,
                    test_sim_label,
                    test_sim_output,
                )

            else:
                if os.path.isfile(self.file_path):
                    print("read data2 ...")
                    file = h5py.File(self.file_path, "r")

                    train_sim_feature = torch.from_numpy(
                        file["train_sim_feature"][:]
                    ).float()
                    
                    train_sim_att_2 = torch.from_numpy(
                        file["train_sim_att_2"][:].squeeze()
                    ).float()
                    train_sim_output = torch.from_numpy(
                        file["train_sim_output"][:]
                    ).float()
                    train_sim_label = torch.from_numpy(file["train_sim_label"][:]).int()
                    test_sim_feature = (
                        torch.from_numpy(file["test_sim_feature"][:]).float().squeeze()
                    )
                    
                    test_sim_att_2 = torch.from_numpy(
                        file["test_sim_att_2"][:].squeeze()
                    ).float()
                    test_sim_label = torch.from_numpy(file["test_sim_label"][:]).int()

                    test_sim_feature = torch.from_numpy(
                        np.array(test_sim_feature)
                    ).float()
                    print("train_sim_feature.shape:", train_sim_feature.shape)
                    print("train_sim_output.shape:", train_sim_output.shape)
                    print("test_sim_feature:", test_sim_feature.shape)
                    print("test_sim_label:", test_sim_label.shape)

                    test_seen_sim_feature = (
                        torch.from_numpy(file["test_seen_sim_feature"][:])
                        .float()
                        .squeeze()
                    )
                    
                    test_seen_sim_att_2 = torch.from_numpy(
                        file["test_seen_sim_att_2"][:].squeeze()
                    ).float()
                    test_seen_sim_label = torch.from_numpy(
                        file["test_seen_sim_label"][:]
                    ).int()

                    print("test_seen_sim_feature:", test_seen_sim_feature.shape)
                    print("test_seen_sim_label:", test_seen_sim_label.shape)
                    file.close()
                    print("read data done.")
                    
                    return (
                        train_sim_feature,
                        train_sim_att_2,
                        train_sim_output,
                        train_sim_label,
                        test_seen_sim_feature,
                        test_seen_sim_att_2,
                        test_seen_sim_label,
                        test_sim_feature,
                        test_sim_att_2,
                        test_sim_label,
                    )
                else:
                    print("prepare data3 ...")
                    cla_feat = {}  # 记录每一个训练类的样本
                    for i in np.unique(self.dataset.train_label.numpy()):
                        cla_feat[i] = self.dataset.train_feature.numpy()[
                            np.where(self.dataset.train_label.numpy() == i)[0]
                        ]

                    # mask住unseen类的attribute，将搜索空间限制在seen类
                    mask_attribute = np.zeros((self.dataset.attribute.shape))
                    for idx, att in enumerate(self.dataset.attribute):
                        mask_attribute[idx] = (
                            att
                            if idx in np.unique(self.dataset.train_label.numpy())
                            else np.zeros(self.dataset.attribute.shape[-1])
                        )
                    class_knn = NearestNeighbors(
                        n_neighbors=opt.k_class + 1, metric="cosine"
                    ).fit(mask_attribute)
                    _, ind = class_knn.kneighbors(self.dataset.attribute)
                    class_dic = {}  # 记录每一个类在train_classes中的相似类的序号,通过attribute寻找相似类
                    for idx, target in enumerate(ind):
                        tar = (
                            target[1:opt.k_class+1]
                            if idx in np.unique(self.dataset.train_label.numpy())
                            else target[0:opt.k_class]
                        )
                        class_dic[idx] = tar

                    # 数据准备
                    # seen_class_samples = {}
                    # for seen_class in np.unique(self.dataset.train_label.numpy()):
                    #     sim_cla_feat = cla_feat[seen_class]  # 类别i的所有样本

                    # 只取靠近样本中心的样本，用于测试阶段的生成。
                    # centers = self.dataset.tr_cls_centroid[seen_class]
                    # diff = (
                    #     np.tile(centers, (sim_cla_feat.shape[0], 1)) - sim_cla_feat
                    # )
                    # distance = np.sum(diff ** 2, axis=1) ** 0.5
                    # max_index = np.argsort(distance)
                    # # 只取该类中，离样本中心最近的n%样本用于生成。
                    # samples_selected = sim_cla_feat[max_index][
                    #     : int(opt.selected_ratio * sim_cla_feat.shape[0])
                    # ]

                    # km_cluster = KMeans(
                    #     n_clusters=opt.k_inst,
                    #     max_iter=300,
                    #     n_init=40,
                    #     init="k-means++",
                    #     n_jobs=-1,
                    # )
                    # # 该类样本按k_inst聚类
                    # clusters = km_cluster.fit_predict(samples_selected)
                    # seen_class_samples[seen_class] = {
                    #     k: samples_selected[np.where(clusters == k)[0]]
                    #     for k in range(opt.k_inst)
                    # }

                    # for k, v in seen_class_samples_sorted.items():
                    #     print(k, v.shape)
                    # 根据距离中心排序得到的seen samples for each class, 词典存放
                    # 在读取时，可以用来根据ratio调整用于生成的seen samples
                    # sio.savemat(os.path.join(
                    #         self.datapath,
                    #         "%s/KnnFeat_%d_%d_attention.mat"%(opt.dataset, opt.k_class, opt.k_inst)
                    #     ), seen_class_samples_sorted)

                    # 生成 train_feature 的输入
                    train_sim_feature = []  # end-to-end input when training
                    train_sim_att_2 = []  # att of output
                    train_sim_output = []  # end-to-end output when training
                    train_sim_label = []

                    for i, j in tqdm(zip(
                            self.dataset.train_feature.numpy(),
                            self.dataset.train_label.numpy(),
                        )):
                        # cla is the most similar class to class j in attribute
                        once_input_feat = np.array([]).reshape(0, 2048)
                        for idx, cla in enumerate(class_dic[j]):  
                            cla_samples = cla_feat[cla]
                            similar_samples = find_similar_samples(
                                i.reshape(1, -1), cla_samples, k=opt.k_inst
                            )
                            for sample in similar_samples:
                                once_input_feat = np.vstack((once_input_feat, sample))
                        train_sim_feature.append(once_input_feat)

                        train_sim_att_2.append(
                            np.array(self.dataset.attribute[j])
                        )  # att2 为需要生成的类的属性。
                        train_sim_output.append(np.array(i))  # source feature
                        train_sim_label.append(j)  # target label

                            # for k in range(opt.k_inst):
                            #     cluster_samples = seen_class_samples[cla][k]
                            #     most_similar_sample_in_cluster = find_most_close_sample(
                            #         i.reshape(1, -1), cluster_samples
                            #     )
                            #     train_sim_feature.append(most_similar_sample_in_cluster)

                            #     # training process: temp -> hidden layer -> generated sample, compare with i
                            #     train_sim_att_1.append(
                            #         np.array(self.dataset.train_att[cla])
                            #     )
                            #     train_sim_att_2.append(
                            #         np.array(self.dataset.train_att[j])
                            #     )  # att2 为需要生成的类的属性。
                            #     train_sim_output.append(np.array(i))  # source feature
                            #     train_sim_label.append(j)  # target label

                    # train_sim_output = self.dataset.train_feature.numpy(),
                    # train_sim_label = self.dataset.train_label.numpy()
                    # train_sim_att_2 = self.dataset.train_att[self.dataset.train_label.numpy()]

                    train_sim_feature = np.array(train_sim_feature).squeeze().astype(float)
                    
                    print("train_sim_feature.shape:", train_sim_feature.shape)
                    train_sim_feature = torch.from_numpy(train_sim_feature)  # label 和 train_label一致
                   
                    train_sim_att_2 = np.array(train_sim_att_2).astype(float)
                    train_sim_att_2 = torch.from_numpy(train_sim_att_2)
                    print("train_sim_att_2.shape:", train_sim_att_2.shape)
                    train_sim_output = (
                        np.array(train_sim_output).squeeze().astype(float)
                    )
                    train_sim_output = torch.from_numpy(train_sim_output)
                    train_sim_label = np.array(train_sim_label).squeeze().astype(int)
                    train_sim_label = torch.from_numpy(train_sim_label)

                    print("first step finished")

                    # 为测试阶段准备数据
                    # 生成 test_seen_feature 的输入
                    test_seen_sim_feature = []
                    test_seen_sim_att_2 = []
                    test_seen_sim_label = []

                    # 用cla生成i
                    for i in np.unique(self.dataset.test_seen_label.numpy()):
                        # 每一类i，产生SYN_NUM个
                        for _ in range(opt.nSample):
                            once_input_feat = np.array([]).reshape(0, 2048)
                            for cla in class_dic[i]:
                                cla_samples = cla_feat[cla]
                                choices = random.sample(list(range(0, len(cla_samples))), opt.k_inst)
                                for c in choices:
                                    once_input_feat = np.vstack((once_input_feat, cla_samples[c]))
                            
                            test_seen_sim_label.append(i)
                            test_seen_sim_att_2.append(np.array(self.dataset.attribute[i]))
                            test_seen_sim_feature.append(once_input_feat)

                    test_seen_sim_feature = (
                        np.array(test_seen_sim_feature).squeeze().astype(float)
                    )
                    print("test_seen_sim_feature.shape:", test_seen_sim_feature.shape)
                    test_seen_sim_feature = torch.from_numpy(test_seen_sim_feature)
                    test_seen_sim_label = np.array(test_seen_sim_label).astype(int)
                    test_seen_sim_label = torch.from_numpy(test_seen_sim_label)
                    test_seen_sim_att_2 = (
                        np.array(test_seen_sim_att_2).astype(float)
                    )
                    test_seen_sim_att_2 = torch.from_numpy(test_seen_sim_att_2)
                    print("test_seen_sim_att_2.shape:", test_seen_sim_att_2.shape)
                    print("second step finished")

                    # 生成 test_unseen_feature 的输入
                    test_sim_feature = []
                    test_sim_att_2 = []
                    test_sim_label = []

                    for i in np.unique(self.dataset.test_unseen_label.numpy()):
                        for _ in range(opt.nSample):
                            once_input_feat = np.array([]).reshape(0, 2048)
                            for cla in class_dic[i]:
                                cla_samples = cla_feat[cla]
                                choices = random.sample(list(range(0, len(cla_samples))), opt.k_inst)
                                for c in choices:
                                    once_input_feat = np.vstack((once_input_feat, cla_samples[c]))
                            
                            test_sim_label.append(i)
                            test_sim_att_2.append(np.array(self.dataset.attribute[i]))
                            test_sim_feature.append(once_input_feat)
                            
                    test_sim_feature = (
                        np.array(test_sim_feature).squeeze().astype(float)
                    )
                    print("test_sim_feature.shape:", test_sim_feature.shape)
                    test_sim_feature = torch.from_numpy(test_sim_feature)
                    test_sim_label = np.array(test_sim_label).astype(int)
                    test_sim_label = torch.from_numpy(test_sim_label)
                    test_sim_att_2 = np.array(test_sim_att_2).squeeze().astype(float)
                    test_sim_att_2 = torch.from_numpy(test_sim_att_2)
                    print("test_sim_att_2.shape:", test_sim_att_2.shape)
                    
                    print("third step finished")
                    # 创建file，方便下次读取
                    file = h5py.File(self.file_path, "w")

                    # file.create_dataset(
                    #     "test_seen_sim_label_source", data=np.array(test_seen_sim_label_source)
                    # )
                    # file.create_dataset(
                    #     "test_sim_label_source", data=np.array(test_sim_label_source)
                    # )
                    #############以上只用于数据读取阶段

                    file.create_dataset(
                        "train_sim_feature", data=train_sim_feature.numpy()
                    )
                    file.create_dataset("train_sim_att_2", data=train_sim_att_2.numpy())
                    file.create_dataset(
                        "train_sim_output", data=train_sim_output.numpy()
                    )
                    file.create_dataset("train_sim_label", data=train_sim_label.numpy())

                    file.create_dataset(
                        "test_sim_feature", data=test_sim_feature.numpy()
                    )
                    file.create_dataset("test_sim_att_2", data=test_sim_att_2.numpy())
                    file.create_dataset("test_sim_label", data=test_sim_label.numpy())

                    file.create_dataset(
                        "test_seen_sim_feature", data=test_seen_sim_feature.numpy()
                    )
                    
                    file.create_dataset(
                        "test_seen_sim_att_2", data=test_seen_sim_att_2.numpy()
                    )
                    file.create_dataset(
                        "test_seen_sim_label", data=test_seen_sim_label.numpy()
                    )
                    file.close()
                    print("prepare data done.")

                    """以AWA为例，label顺序重新整理：test_seen_sim_label：0-39；test_sim_label：40-49
                    """
                    return self._getKnnFeat(opt)


class SVM(object):
    def __init__(self, tr_x, tr_y):
        self.classifier = LinearSVC()
        self.classifier.fit(tr_x, tr_y)

    def acc(self, tt_x, tt_y):
        predict = self.classifier.predict(tt_x)
        acc = np.zeros(len(np.unique(tt_y)))
        for i, j in zip(range(tt_y.max() + 1), np.unique(tt_y)):
            acc[i] = (predict[tt_y == j] == j).mean()
        acc = acc.mean() * 100

        return acc


class Softmax(object):
    def __init__(self, tr_x, tr_y):
        self.n_cla = len(np.unique(tr_y))
        self.classifier = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", C=self.n_cla, max_iter=200
        )
        self.classifier.fit(tr_x, tr_y.squeeze())

    def acc(self, tt_x, tt_y):
        tt_y = tt_y.squeeze()
        predict = self.classifier.predict(tt_x)

        acc = np.zeros(len(np.unique(tt_y)))
        # print(predict[:100])
        # print(tt_y[:100])
        for i, j in zip(range(tt_y.max() + 1), np.unique(tt_y)):
            acc[i] = (predict[tt_y == j] == j).mean()  # 取每个类的平均准确率，再平均
        acc = acc.mean() * 100

        acc2 = (predict == tt_y).mean() * 100
        return acc, acc2


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias, 0.0)


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(input_dim/2), nclass),
            nn.ReLU(),
        )

    def forward(self, x):
        o = self.fc(x)
        return o


class Classifier:
    def __init__(
        self, tr_x, tr_y, _nclass, _nepoch=25, _batch_size=256, _lr=0.001, _beta1=0.5
    ):
        self.nclass = _nclass
        self.input_dim = tr_x.shape[1]
        self.ntrain = tr_x.shape[0]
        self.nepoch = _nepoch
        self.batch_size = _batch_size
        self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.model.apply(weights_init)
        self.criterion = nn.CrossEntropyLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=_lr, betas=(_beta1, 0.999)
        )
        self.loader = DataLoader(
            dataset=TensorDataset(tr_x, tr_y),
            batch_size=_batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        self.train()

    def train(self):
        with torch.set_grad_enabled(True):
            self.model.cuda()
            self.model.train()
            for epoch in range(self.nepoch):
                for batch_input, batch_label in self.loader:
                    self.optimizer.zero_grad()
                    # batch_input, batch_label = self.next_batch(self.batch_size)
                    self.input.copy_(batch_input)
                    self.label.copy_(batch_label)

                    inputv = Variable(self.input).cuda()
                    labelv = Variable(self.label).cuda()
                    output = self.model(inputv)
                    loss = self.criterion(output, labelv)
                    loss.backward()
                    self.optimizer.step()

    def acc(self, tt_x, tt_y):
        with torch.no_grad():
            self.model.eval()
            tt_y = tt_y.squeeze()
            _, predict = torch.max(self.model(tt_x.cuda()).detach().cpu(), 1)
            # predict = predict.numpy()
            acc = []
            # print(predict[:100])
            # print(tt_y[:100])
            for i in range(self.nclass):
                pred = predict[tt_y == i]
                if torch.sum(pred) == 0:
                    continue
                acc.append(
                    (torch.sum(pred == i).float() / pred.shape[0]).numpy()
                )  # 取每个类的平均准确率，再平均
            # print(acc)
            acc = np.mean(acc) * 100
            acc2 = torch.sum(predict == tt_y).float() / tt_y.shape[0] * 100
            return acc, acc2

