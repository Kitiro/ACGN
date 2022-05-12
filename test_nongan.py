import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn
import os
from dataset import DATA_LOADER, TestKnn, SVM, Softmax
from models import _netCC_3, _param
import argparse
import numpy as np 


parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--gpu', type=str, default='-1')
parser.add_argument('--dataset', dest='dataset', type=str, default='AWA1')
parser.add_argument('--dataroot', default='/data/liujinlu/xian_resnet101/xlsa17/data/',
                    help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--split', type=str, default='Nongan')
parser.add_argument('--batchsize', type = int, default = 5, help= 'batchsize')

opt = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu



def test_zsl(datadir, test_input, netG, opt, dataset):
    acclog = ''
    keys = test_input.unseen_source_input.keys()
    with open(datadir+'/test_zsl_acc.txt','w') as file:
        for key in keys:
            test_sim_feature, test_sim_att_2, test_sim_label = test_input.unseen_source_input[key].numpy(), test_input.unseen_target_att[key].numpy(), test_input.unseen_target_label[key].numpy()
            cur = 0
            G_sample = np.zeros([0, 2048])
            while(True):
                if cur+opt.batchsize >= len(test_sim_feature):
                    temp_x = test_sim_feature[cur:cur+opt.batchsize]
                    temp_att = test_sim_att_2[cur:cur+opt.batchsize]
                    temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
                    temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
                    g_sample, _ = netG(temp_x, temp_att)
                    G_sample = np.vstack((G_sample, g_sample.detach().cpu().numpy()))
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

            print("KEYS: {}, Accuracy is {:.2f}%".format(key, acc))
            acclog = "KEYS: {}, Accuracy is {:.2f}%".format(key, acc)+'\n'
            file.write(acclog)

def test_gzsl(datadir, test_input, netG, opt, dataset):
    acclog = ''
    # test_unseen
    keys = test_input.unseen_source_input.keys()
    with open(datadir+'/test_gzsl_acc.txt','w') as file:
        for key in keys:
            test_sim_feature, test_sim_att_2, test_sim_label = test_input.unseen_source_input[key].numpy(), test_input.unseen_target_att[key].numpy(), test_input.unseen_target_label[key].numpy()
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
            test_seen_sim_feature, test_seen_sim_att_2, test_seen_sim_label = test_input.seen_source_input[key].numpy(), test_input.seen_target_att[key].numpy(), test_input.seen_target_label[key].numpy()
            cur = 0
            G_seen_sample = np.zeros([0, 2048])
            while(True):
                if cur+opt.batchsize >= len(test_seen_sim_feature):
                    temp_x = test_seen_sim_feature[cur:cur+opt.batchsize]
                    temp_att = test_seen_sim_att_2[cur:cur+opt.batchsize]
                    temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
                    temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
                    g_sample, _ = netG(temp_x, temp_att)
                    G_seen_sample = np.vstack((G_seen_sample, g_sample.detach().cpu().numpy()))
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

            print("KEYS: {}, H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(key, acc, acc_S_T, acc_U_T))
            acclog = "KEYS: {}, H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(key, acc, acc_S_T, acc_U_T) + '\n'
            file.write(acclog)

if __name__ == '__main__':
    dataset = DATA_LOADER(opt)
    test_input = TestKnn(opt)

    if opt.gpu!='-1':
        netG = _netCC_3(dataset.train_att.numpy().shape[1]).cuda()
    else:
        netG = _netCC_3(dataset.train_att.numpy().shape[1])

    opt.path_root = '/data/liujinlu/zsl/Cross_Class_GAN/data/%s/%s' %(opt.dataset, opt.split)
    for file in os.listdir(opt.path_root):
        for model in os.listdir(os.path.join(opt.path_root, file)):
            datadir = os.path.join(opt.path_root, file)
            if model[:14]=='Best_model_ZSL':
                zsl_model = torch.load(os.path.join(opt.path_root, file, model))
                netG.load_state_dict(zsl_model['state_dict_G'])
                netG.eval()
                test_zsl(datadir, test_input, netG, opt, dataset)

            if model[:15]=='Best_model_GZSL':
                gzsl_model = torch.load(os.path.join(opt.path_root, file, model))
                netG.load_state_dict(gzsl_model['state_dict_G'])
                netG.eval()
                test_gzsl(datadir, test_input, netG, opt, dataset)







