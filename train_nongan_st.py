# coding:utf8
# 毕设将bmvc：cross_gan改为对抗式和非对抗式
# 本脚本为非对抗式
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from termcolor import cprint
from time import gmtime, strftime
from sklearn.neighbors import NearestNeighbors
import numpy as np
import argparse
import os
import glob
import random
import json
import h5py
import dateutil.tz
import datetime
from scipy import io as sio

#from tensorboardX import SummaryWriter

from dataset import FeatDataLayer, DATA_LOADER, KnnFeat, SVM, Softmax, Classifier
from models import _netCC_3, _param, _netCC_3_3

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AwA2')
parser.add_argument('--dataroot', default='data',
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
parser.add_argument('--att_w', type = float, default = 0.1, help= 'weight of Att_loss')
parser.add_argument('--x_w', type = float, default = 0.1, help= 'weight of X_loss')
parser.add_argument('--cent_w', type = float, default = 0.05, help= 'weight of X_loss')
parser.add_argument('--s_w', type = float, default = 5, help= 'weight of Semantic_loss')
parser.add_argument('--u_w', type = float, default = 1, help= 'weight of Semantic_loss for Unseen class')

opt = parser.parse_args()
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.GP_LAMBDA = 10    # Gradient penalty lambda
opt.CENT_LAMBDA  = opt.cent_w  #Weight of Centroid loss 
opt.REG_W_LAMBDA = 0.001 #Weight of reg loss 
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
opt.path_root = 'data/%s' %opt.dataset



if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)



def train():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    out_dir  = opt.path_root + '/Nongan'
    if not opt.random:
        #out_subdir = opt.path_root + '/Nongan/Kcla{:d}_Kinst{:d}_{:s}'.format(opt.k_class,opt.k_inst,timestamp)
        out_subdir = opt.path_root + '/Nongan/Kcla{:d}_Kinst{:d}_Alpha{}_Beta{}_Gamma{}'.format(opt.k_class,opt.k_inst,opt.att_w, opt.x_w, opt.cent_w)
    else:
        out_subdir = opt.path_root + '/Nongan/Random_{:s}'.format(timestamp)

    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # if not os.path.exists(out_subdir):
    #     os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir = out_subdir + '/log.txt'
    # with open(log_dir, 'w') as f:
    #     f.write('Training Start:')
    #     f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    #     for k, v in zip(vars(opt).keys(),vars(opt).values()):
    #         f.write(k+':'+str(v))
    # summary_writer = SummaryWriter(out_subdir)


    Tensor = torch.cuda.FloatTensor
    param = _param()
    dataset = DATA_LOADER(opt)

    preinputs = KnnFeat(opt)
    param.X_dim = dataset.feature_dim
    
    # prepare data for input
    # 映射后的semantic feature
    mapping_semantic = torch.from_numpy(sio.loadmat('data/AwA2/mapped_semantic.mat')['mapped_feature'])
    mapping_semantic_seen = mapping_semantic[dataset.seenclasses]
    #print(preinputs.test_sim_label)
    #print(preinputs.test_sim_label.shape)
    mapping_semantic_unseen = mapping_semantic[preinputs.test_sim_label.long()] # 产生的unseen类的label对应的mapped semantic feature
    print('mapping feature shape:', mapping_semantic_unseen.shape)
     
    # generated_unseen_x = preinputs.test_sim_feature # 用于产生unseen的seen feature
    # generated_unseen_att = preinputs.test_sim_att_2
    # generated_unseen_label = preinputs.test_sim_label.long() # 40 - 49

    # concatenate seen and unseen for training
    print(preinputs.train_sim_label.shape)
    print(preinputs.test_sim_label.shape)
    train_label = np.concatenate((preinputs.train_sim_label.numpy(), preinputs.test_sim_label.numpy()))
    print(train_label.shape)
    feat_data = np.concatenate((preinputs.train_sim_feature.numpy(), mapping_semantic_unseen))  # target feature for MSE loss
    x = np.concatenate((preinputs.train_sim_feature.numpy(), preinputs.test_sim_feature.numpy()))  # source feature
    att1 = np.concatenate((preinputs.train_sim_att_1.numpy(), preinputs.test_sim_att_1.numpy()))  # source att
    att2 = np.concatenate((preinputs.train_sim_att_2.numpy(), preinputs.test_sim_att_2.numpy()))  # target att
    
    data_layer = FeatDataLayer(train_label, feat_data, x, att1, att2, opt)
    
    #print('train_label:', np.unique(preinputs.train_sim_label.numpy()))
    result = Result()
    result_gzsl = Result()

    netG = _netCC_3(preinputs.train_sim_att_2.numpy().shape[1]).cuda()
    print('att_dim:', preinputs.train_sim_att_2.numpy().shape[1])
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

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).cuda()  # centers for each seen class
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    

    for it in range(start_step, opt.endEpoch):
        #print('epoch: ',it)

        """ Generator """
        for _ in range(opt.G_epoch):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # target class sample
            x = blobs['x'] # source class sample
            att1 = blobs['att1']
            att = blobs['att2'] # target class attribute
            labels = blobs['labels'].astype(int)  # target class labels
            
            x = Variable(torch.from_numpy(x.astype('float32'))).cuda()
            att = Variable(torch.from_numpy(att.astype('float32'))).cuda()
            att1 = Variable(torch.from_numpy(att1.astype('float32'))).cuda()
            # valid = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)

            X = Variable(torch.from_numpy(feat_data.astype('float32'))).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()

            # generated samples and hidden layers output
            G_sample, G_att = netG(x, att)

            # GAN's att loss
            Att_loss = opt.att_w * F.mse_loss(G_att, att)

            # GAN's x loss
            X_loss = opt.x_w * F.mse_loss(G_sample, X)

            # # Centroid loss
            Euclidean_loss_target = Variable(torch.Tensor([0.0])).cuda()

            # generated visual and mapping attribute loss. 
            # mapping_semantic_batch = mapping_semantic_seen[labels].cuda()

            #semantic_loss = opt.s_w * F.mse_loss(G_sample, mapping_semantic_batch)
            
            if opt.REG_W_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze() #  sample belong to class i in this batch
                    if sample_idx.numel() == 0:  # return the quantity
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
            
            # unseen_G_sample loss; temp_x and temp_att 是我们需要产生的unseen sample对应的source visual feature 和target attribute
            # G_sample, _ = netG(temp_x, temp_att)
            # #G_sample = G_sample.detach().cpu().numpy()
            # u_loss = opt.u_w * F.mse_loss(G_sample, mapping_semantic_unseen.clone().cuda())
            
            all_loss = Att_loss + X_loss + Euclidean_loss_target + reg_loss #+ semantic_loss + u_loss
            #all_loss = reg_loss +  X_loss + Euclidean_loss_target
            all_loss.backward()

            optimizerG.step()
            # summary_writer.add_scalar('G_loss', all_loss.item(), it)
            reset_grad(nets)


        if it % opt.disp_interval == 0 and it:

            #log_text = 'Iter-{}; Euclidean_loss_target:{:.3f};reg_loss:{:.3f}; att_loss:{:.3f}; x_loss:{:.3f}; Semantic_loss:{:.3f}; Unseen_loss:{:.3f};'.format(it, Euclidean_loss_target.item(),reg_loss.item(),Att_loss.item(), X_loss.item(), semantic_loss.item(), u_loss.item())
            log_text = 'Iter-{}; Euclidean_loss_target:{:.3f};reg_loss:{:.3f}; x_loss:{:.3f}'.format(it, Euclidean_loss_target.item(),reg_loss.item(), X_loss.item())
            print(log_text)
        #     with open(log_dir, 'a') as f:
        #         f.write(log_text+'\n')

        if it % opt.evl_interval == 0 and it >= 40:
            netG.eval()
            eval_fakefeat_test(it, netG, preinputs, opt, result, dataset)
            if result.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model_ZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                
                # save_model(it, netG, opt.manualSeed, log_text,
                #            out_subdir + '/Best_model_ZSL_Acc_{:.2f}_Epoch_{:d}.tar'.format(result.acc_list[-1], it))
                # with open('data/AwA2/Nongan/results.txt', 'a') as out:
                #     out.write('Acc_{:.2f}_Epoch_{:d}\n'.format(result.acc_list[-1], it))
                # print('Acc_{:.2f}_Epoch_{:d}\n'.format(result.acc_list[-1], it))
            if opt.gzsl: 
                eval_fakefeat_test_gzsl(it, netG, preinputs, opt, result_gzsl, dataset)

                if result_gzsl.save_model:
                    files2remove = glob.glob(out_subdir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    
                    # save_model(it, netG, opt.manualSeed, log_text,
                    #            out_subdir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}_Epoch_{:d}.tar'.format(result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T,it))
                print('Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}_Epoch_{:d}.tar'.format(result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T,it))   
            netG.train()
    # with open('results/nongan_result.txt','a') as out:
    #     out_info = 'Kcla{:d}_Kinst{:d}_Alpha{}_Beta{}_Acc{}\n'.format(opt.k_class,opt.k_inst,opt.att_w, opt.x_w, result.acc_list[-1])
    #     out.write(out_info)

    with open(f'data/AwA2/Nongan/Kcla{opt.k_class}_Kinst{opt.k_inst}_self_traing_results.txt', 'a') as out:
        out.write('Att_w:{:.3f}_X_w:{:.3f}_Centroid_w:{:.3f}_Semantic_w:{:.3f}_Unseen_w:{:.3f}\nAcc:{}_StdAcc:{}\n'.format(opt.att_w, opt.x_w, opt.cent_w, opt.s_w, opt.u_w, result.best_acc, result.best_acc_std))


def save_model(it, netG, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


# 每个batch尝试生成一次unseen sample，利用mapping semantic feature检验生成质量和区分度


def eval_fakefeat_test(it, netG, preinputs, opt, result, dataset):
    test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label = preinputs.test_sim_feature.numpy(), preinputs.test_sim_att_1.numpy(), preinputs.test_sim_att_2.numpy(), preinputs.test_sim_label.numpy()


    temp_x = Variable(torch.from_numpy(test_sim_feature.astype('float32'))).cuda()
    temp_att = Variable(torch.from_numpy(test_sim_att_2.astype('float32'))).cuda()
    G_sample, _ = netG(temp_x, temp_att)
    G_sample = G_sample.detach().cpu().numpy()
        
    # cur = 0
    # G_sample = np.zeros([0, 2048])
    # while(True):
    #     if cur+opt.batchsize >= len(test_sim_feature):
    #         temp_x = test_sim_feature[cur:cur+opt.batchsize]
    #         temp_att1 = test_sim_att_1[cur:cur+opt.batchsize]
    #         temp_att = test_sim_att_2[cur:cur+opt.batchsize]
            
    #         temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
    #         temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
    #         temp_att1 = Variable(torch.from_numpy(temp_att1.astype('float32'))).cuda()

    #         g_sample, _ = netG(temp_x, temp_att)
    #         G_sample = np.vstack((G_sample, g_sample.numpy()))
    #         cur = cur + opt.batchsize
    #     else:
    #         temp_x = test_sim_feature[cur:]
    #         temp_att1 = test_sim_att_1[cur:]
    #         temp_att = test_sim_att_2[cur:]
            
    #         temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
    #         temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
    #         temp_att1 = Variable(torch.from_numpy(temp_att1.astype('float32'))).cuda()
            
    #         g_sample, _ = netG(temp_x, temp_att)
    #         G_sample = np.vstack((G_sample, g_sample.detach().cpu().numpy()))
    #         break
    
    unseen_label = []
    for i in dataset.test_unseen_label.numpy():
        unseen_label.append(i+dataset.ntrain_class)
    unseen_label = np.array(unseen_label)

    # classifier = SVM(G_sample, test_sim_label)
    classifier = Classifier(G_sample, test_sim_label-dataset.ntrain_class, _nclass = dataset.ntest_class)
    acc, acc2 = classifier.acc(dataset.test_unseen_feature, dataset.test_unseen_label.numpy())
    
    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True
    if acc2 > result.best_acc_std:
        result.best_acc_std = acc2
        result.best_iter = it
        result.save_model = True
    best = result.best_acc if result.best_acc > result.best_acc_std else result.best_acc_std
    print("Accuracy is {:.2f}%, standard acc : {:.2f}%, best acc : {:.2f}%".format(acc, acc2, best))


def eval_fakefeat_test_gzsl(it, netG, preinputs, opt, result, dataset):
    # test_unseen
    test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label = preinputs.test_sim_feature.numpy(), preinputs.test_sim_att_1.numpy(), preinputs.test_sim_att_2.numpy(), preinputs.test_sim_label.numpy()
    cur = 0
    G_unseen_sample = np.zeros([0, 2048])
    while(True):
        if cur+opt.batchsize >= len(test_sim_feature):
            temp_x = test_sim_feature[cur:cur+opt.batchsize]
            temp_att = test_sim_att_2[cur:cur+opt.batchsize]
            temp_att1 = test_sim_att_1[cur:cur+opt.batchsize]
            
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            temp_att1 = Variable(torch.from_numpy(temp_att1.astype('float32'))).cuda()
            
            g_sample, _ = netG(temp_x, temp_att)
            G_unseen_sample = np.vstack((G_unseen_sample, g_sample.numpy()))
            cur = cur + opt.batchsize
        else:
            temp_x = test_sim_feature[cur:]
            temp_att1 = test_sim_att_1[cur:]
            temp_att = test_sim_att_2[cur:]
            
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            temp_att1 = Variable(torch.from_numpy(temp_att1.astype('float32'))).cuda()
            
            g_sample, _ = netG(temp_x, temp_att)
            G_unseen_sample = np.vstack((G_unseen_sample, g_sample.detach().cpu().numpy()))
            break

    # test_seen
    test_seen_sim_feature, test_seen_sim_att_1, test_seen_sim_att_2, test_seen_sim_label = preinputs.test_seen_sim_feature.numpy(), preinputs.test_seen_sim_att_1.numpy(), preinputs.test_seen_sim_att_2.numpy(), preinputs.test_seen_sim_label.numpy()
    cur = 0
    G_seen_sample = np.zeros([0, 2048])
    while(True):
        if cur+opt.batchsize >= len(test_seen_sim_feature):
            temp_x = test_seen_sim_feature[cur:cur+opt.batchsize]
            temp_att = test_seen_sim_att_2[cur:cur+opt.batchsize]
            temp_att1 = test_seen_sim_att_1[cur:cur+opt.batchsize]
            
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            temp_att1 = Variable(torch.from_numpy(temp_att1.astype('float32'))).cuda()
            g_sample, _ = netG(temp_x, temp_att)
            G_seen_sample = np.vstack((G_seen_sample, g_sample.numpy()))
            cur = cur + opt.batchsize
        else:
            temp_x = test_seen_sim_feature[cur:]
            temp_att = test_seen_sim_att_2[cur:]
            temp_att1 = test_seen_sim_att_1[cur:]
            
            temp_x = Variable(torch.from_numpy(temp_x.astype('float32'))).cuda()
            temp_att = Variable(torch.from_numpy(temp_att.astype('float32'))).cuda()
            temp_att1 = Variable(torch.from_numpy(temp_att1.astype('float32'))).cuda()
            
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
    acc_S_T, acc_S_T2 = classifier.acc(dataset.test_seen_feature.numpy(), dataset.test_seen_label.numpy())

    # U-->T
    acc_U_T, acc_U_T2 = classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)

    acc = (2 * acc_S_T * acc_U_T) / (acc_S_T + acc_U_T)
    acc2 = (2 * acc_S_T2 * acc_U_T2) / (acc_S_T2 + acc_U_T2)
    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.best_acc_S_T = acc_S_T
        result.best_acc_U_T = acc_U_T
        result.save_model = True
    if acc2 > result.best_acc_std:
        result.best_acc_std = acc2
        result.best_iter = it
        result.save_model = True
    out_log = "H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  StdAcc:{} \n Best Acc: {:.2f}\t Best Acc_Std:{:.2f}\n".format(acc, acc_S_T, acc_U_T, acc2, result.best_acc, result.best_acc_std)
    out_file = f'data/AwA2/Nongan/Kcla{opt.k_class}_Kinst{opt.k_inst}_Att_w:{opt.att_w}_X_w:{opt.x_w}_Centroid_w:{opt.cent_w}_Semantic_w:{opt.s_w}_self_traing_results_gzsl.txt'
    print(out_log)
    with open(out_file, 'a') as out:
        out.write(out_log)
        
class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_acc_std = 0.0
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



if __name__ == "__main__":
    train()

