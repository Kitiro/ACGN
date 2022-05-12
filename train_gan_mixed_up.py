# coding:utf8
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
import numpy as np
import argparse
import os
import glob
import random
import json
import h5py
import dateutil.tz
import datetime
from tqdm import tqdm

from dataset_for_attention import FeatDataLayer, DATA_LOADER, KnnFeat_attention, Classifier, map_label, Softmax
from models import  _param, _netD_SC, _netCC_attention_multi_head, _netCC_3_self_attention

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

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--exp_idx', default='', type=str, help='exp idx')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume',  type=str, help='the model to resume')

parser.add_argument('--x_dim',  type=int, default=2048)
parser.add_argument('--att_dim',  type=int, default=85)
parser.add_argument('--z_dim',  type=int, default=300, help='dimension of the random vector z')
parser.add_argument('--disp_interval', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=300)
parser.add_argument('--evl_interval',  type=int, default=40)

parser.add_argument('--random', action='store_true', default=False, help = 'random pairs in data preparation')
parser.add_argument('--endEpoch', type = int, default = 5000, help= 'train epoch')
parser.add_argument('--gzsl', action='store_true', default = False, help = 'gzsl evaluation')
parser.add_argument('--batchsize', type = int, default = 1024, help= 'batchsize')
parser.add_argument('--k_class', type = int, default = 1, help= 'find k similar classes')
parser.add_argument('--k_inst', type = int, default = 4, help= 'find k similar instances in each similar class')
parser.add_argument('--att_w', type = int, default = 1, help= 'weight of Att_loss')
parser.add_argument('--x_w', type = int, default = 10, help= 'weight of X_loss')
parser.add_argument('--lr_g', type = float, default = 0.0001, help= 'learning rate of generator')
parser.add_argument('--lr_d', type = float, default = 0.0002, help= 'learning rate of discriminator')
parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value for disc. weights")
parser.add_argument('--cent_w', type = float, default = 3, help= 'CENT_LADMBDA')
parser.add_argument('--extended_attr_num', type = int, default = 0)
parser.add_argument('--selected_ratio', type = float, default = 0.5, help='Select part of training samples for generation.')
parser.add_argument('--split_num', type = int, default = 8, help='Split the feature into several parts.')
parser.add_argument('--save_model', action='store_true', default=False)


opt = parser.parse_args()
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
np.seterr(divide='ignore',invalid='ignore')

""" hyper-parameter  """
opt.GP_LAMBDA = 5    # Gradient penalty lambda
opt.CENT_LAMBDA  = opt.cent_w
opt.REG_W_LAMBDA = 0.0001
opt.D_score = 1
opt.D_cls = 10

#opt.lr = 0.0001

""" hyper-parameter for testing"""
opt.nSample = 400  # number of fake feature for each class
opt.Knn = 20      # knn: the value of K
opt.in_features = 600
opt.out_features = 2048
opt.G_epoch = 1
opt.D_epoch = 1
opt.t = 0.01 # 计算adj: e**(-np.linalg.norm(s1-s2)/t)
opt.path_root = 'data/'

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)


def train():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    out_dir  = opt.path_root + opt.dataset + '/gan'
    if not opt.random:
        out_subdir = opt.path_root + opt.dataset + '/gan/Kcla{:d}_Kinst{:d}_{:s}'.format(opt.k_class, opt.k_inst, timestamp)
        # out_subdir = opt.path_root + '/gan/Kcla{:d}_Kinst{:d}_{:s}'.format(opt.k_class,opt.k_inst,timestamp)
    else:
        out_subdir = opt.path_root + opt.dataset + '/gan/Random_{:s}'.format(timestamp)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # cprint(" The output dictionary is {}".format(out_subdir), 'red')
    # log_dir = out_subdir + '/log.txt'
    # with open(log_dir, 'w') as f:
    #     f.write('Training Start:')
    #     f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    #     for k, v in zip(vars(opt).keys(),vars(opt).values()):
    #         f.write(k+':'+str(v)+'\t')
    
    param = _param()
    dataset = DATA_LOADER(opt)
    preinputs = KnnFeat_attention(opt)
    # preinputs = KnnFeat(opt)
    param.X_dim = dataset.feature_dim
    opt.x_dim = dataset.feature_dim
    opt.att_dim = dataset.att_dim
    
    if opt.random:
        data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.numpy(), preinputs.train_sim_feature.numpy(), preinputs.train_sim_att_2.numpy(), opt)
    else:
        data_layer = FeatDataLayer(preinputs.train_sim_label.numpy(), preinputs.train_sim_output.numpy(), preinputs.train_sim_feature.numpy(), preinputs.train_sim_att_2.numpy(), opt)
    
    result = Result()
    result_gzsl = Result()
    # netG = _netCC_3(preinputs.train_sim_att_2.numpy().shape[1]).cuda()
    
    # netG = _netCC_attention(opt).cuda()
    netG = _netCC_3_self_attention(opt).cuda()
    # netG = torch.nn.DataParallel(netG)
    netG.apply(weights_init)
    
    netD = _netD_SC(dataset.train_cls_num+dataset.test_cls_num, dataset.feature_dim).cuda()
    # netD = torch.nn.DataParallel(netD)
    netD.apply(weights_init)

    start_step = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            netD.load_state_dict(checkpoint['state_dict_D'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    nets = [netG, netD]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g)
    # optimizerD = optim.SGD(netD.parameters(), lr=opt.lr_d, betas=(0.5, 0.9))
    # optimizerG = optim.SGD(netG.parameters(), lr=opt.lr_g, betas=(0.5, 0.9))
    # optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr_d, alpha=0.99, eps=1e-08)
    # optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr_g, alpha=0.99, eps=1e-08)      

    D_criterion = nn.CrossEntropyLoss().cuda()
    mse_loss = nn.MSELoss().cuda()
    best_acc = 0.0

    for it in range(start_step, opt.endEpoch):
        # print('epoch: ',it)
        """ Discriminator """
        netD.train()
        netG.eval()
        for _ in range(opt.D_epoch):
            
            blobs = data_layer.forward()
            feat_data = blobs['data'] # target class sample
            x = blobs['x'] # source class sample
            # att1 = blobs['att1']
            att = blobs['att2'] # target class attribute
            labels = blobs['labels'].astype(int)  # true class labels

            x = Variable(torch.from_numpy(x.astype('float32'))).cuda()
            att = Variable(torch.from_numpy(att.astype('float32'))).view(-1, opt.att_dim).cuda()
            X = Variable(torch.from_numpy(feat_data.astype('float32'))).view(-1, opt.x_dim).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).view(-1).cuda()

            # GAN's D loss 
            D_real, C_real = netD(X) # D_real: 1 dim score, C_real: predict class
            D_score_real_loss = torch.mean(D_real) 
            G_sample, _ = netG(x, att) # generated sample, hidden state
            D_fake, C_fake = netD(G_sample)
            D_score_fake_loss = torch.mean(D_fake)
            
            D_score_loss = opt.D_score*(-D_score_real_loss + D_score_fake_loss)
            
            D_cls_loss_real = opt.D_cls*D_criterion(C_real, y_true)
            D_cls_loss_fake = opt.D_cls*D_criterion(C_fake, y_true)
            D_cls_loss = D_cls_loss_real + D_cls_loss_fake

            # train with gradient penalty (WGAN_GP)
            grad_penalty = calc_gradient_penalty(netD, X.data, G_sample.data)

            D_loss = (D_score_loss + D_cls_loss + grad_penalty)
            optimizerD.zero_grad()
            D_loss.backward()
            Wasserstein_D = D_score_real_loss - D_score_fake_loss
            optimizerD.step()
            
            # Clip weights of discriminator
            for p in netD.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
                
            # reset_grad(nets)
            
        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            
            log_text = 'Iter-{}_D; Was_D:{:.3f}; D_score_real_loss: {:.3f}; D_score_fake_loss: {:.3f}; D_cls_loss_real: {:.3f}; D_score_loss: {:.3f}; D_cls_loss_fake: {:.3f}; gradient_penalty: {:.3f}; D_loss: {:.3f}; rl: {:.2f}%; fk: {:.2f}%; '\
                        .format(it, Wasserstein_D.item(), D_score_real_loss.item(), D_score_fake_loss.item(),
                                D_cls_loss_real.item(), D_score_loss.item(), D_cls_loss_fake.item(), grad_penalty.item(), D_loss.item(), acc_real * 100, acc_fake * 100)
            print(log_text)
            
        """ Generator """
        netD.eval()
        netG.train()
        for _ in range(opt.G_epoch):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            x = blobs['x'] # 
            att = blobs['att2']
            labels = blobs['labels'].astype(int)  # class labels

            x = Variable(torch.from_numpy(x.astype('float32'))).cuda()
            att = Variable(torch.from_numpy(att.astype('float32'))).view(-1, opt.att_dim).cuda()

            X = Variable(torch.from_numpy(feat_data.astype('float32'))).view(-1, opt.x_dim).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).view(-1).cuda()
            G_sample, G_att = netG(x, att)
            _, C_real = netD(X)
            D_fake, C_fake = netD(G_sample)
            D_score_fake_loss = opt.D_score*torch.mean(D_fake)
            
            # GAN's G loss
            # Auxiliary classification loss
            GC_loss = opt.D_cls*D_criterion(C_fake, y_true) / 2
            # G_loss = -D_score_fake_loss + GC_loss

             # GAN's att loss
            Att_loss = opt.att_w * F.mse_loss(G_att, att)
            # GAN's x loss
            X_loss = opt.x_w * F.mse_loss(G_sample, X.view(-1, opt.x_dim))

            # 每个batch中的各个类得到的G_sample中心与基于所有样本得到的中心点计算loss
            Euclidean_loss_target = Variable(torch.Tensor([0.0])).cuda()

            if opt.CENT_LAMBDA != 0:
                batch_cls = torch.unique(y_true)
                for i in batch_cls:
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss_target += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        # Euclidean_loss_target += F.mse_loss(G_sample_cls.mean(dim=0), tr_cls_centroid[i]) * opt.CENT_LAMBDA
                        Euclidean_loss_target += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt() * opt.CENT_LAMBDA
                        
                Euclidean_loss_target *= 1.0/batch_cls.shape[0]
                
            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)
            
            all_loss = -D_score_fake_loss + GC_loss + reg_loss + X_loss + Euclidean_loss_target + Att_loss
            optimizerG.zero_grad()
            all_loss.backward()
            optimizerG.step()

        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            
            log_text = 'Iter-{}_G; Euclidean_loss_target:{:.3f};reg_loss:{:.3f}; GD_score_fake_loss: {:.3f}; GC_loss:{:.3f}; x_loss:{:.3f}; att_loss:{:.3f}; G_all_loss:{:.3f}; rl: {:.2f}%; fk: {:.2f}%; '\
                        .format(it, Euclidean_loss_target.item(), reg_loss.item(), D_score_fake_loss.item(), GC_loss.item(), X_loss.item(), Att_loss.item(), all_loss.item(), acc_real * 100, acc_fake * 100)
            print(log_text)
            # with open(log_dir, 'a') as f:
            #     f.write(log_text+'\n')

        if it % opt.evl_interval == 0 and it >= 40:
            netG.eval()
            eval_fakefeat_test(it, netG, preinputs, opt, result, dataset)
            if result.save_model and opt.save_model:
                if not os.path.exists(out_subdir):
                    os.mkdir(out_subdir)
                files2remove = glob.glob(out_subdir + '/Best_model_ZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                save_model(it, netG, netD, opt.manualSeed, log_text,
                           out_subdir + '/Best_model_ZSL_Acc_{:.2f}_Epoch_{:d}.tar'.format(result.acc_list[-1],it))
            if opt.gzsl:
                eval_fakefeat_test_gzsl(it, netG, preinputs, opt, result_gzsl, dataset)
                if result_gzsl.save_model and opt.save_model:
                    files2remove = glob.glob(out_subdir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    # best_acc_gzsl = result.acc_list[-1]
                    # save_model(it, netG, netD, opt.manualSeed, log_text,
                    #            out_subdir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}_Epoch_{:d}.tar'.format(result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T,it))

            netG.train()
    result_file = 'results/attention_result_'+opt.dataset+'.txt'
    with open(result_file,'a') as out:
        out_info = ''
        out_info = 'Kcla{:d}_Kinst{:d}_Att_w:{}_x_w:{}_Cent_W:{}_Selected_ratio:{}_Split_num:{}_Seed:{}_Acc:{}_Acc2:{}\n'.format(opt.k_class,opt.k_inst,opt.att_w, opt.x_w, opt.cent_w, opt.selected_ratio, opt.split_num, opt.manualSeed, result.best_acc, result.best_acc_std)
        if opt.gzsl:
            out_info += 'GZSL_H_STD_{:.2f}_S2_{:.2f}_U2_{:.2f}_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}\n'.format(result_gzsl.best_acc_std, result_gzsl.best_acc_S_T_std, result_gzsl.best_acc_U_T_std, result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T)
            
        out.write(out_info)
        

def save_model(it, netG, netD, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'state_dict_D': netD.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


def eval_fakefeat_test(it, netG, preinputs, opt, result, dataset):
    test_sim_feature, test_sim_att_2, test_sim_label = preinputs.test_sim_feature, preinputs.test_sim_att_2.view(-1, opt.att_dim), preinputs.test_sim_label.view(-1)
    netG.eval()
    G_sample = torch.randn(0).reshape(-1, opt.x_dim)
    cur = 0
    while(True):
        if cur+opt.batchsize <= len(test_sim_feature):
            temp_x = test_sim_feature[cur:cur+opt.batchsize]
            temp_att = test_sim_att_2[cur:cur+opt.batchsize].cuda()
            g_sample, _ = netG(temp_x.cuda(), temp_att)
            G_sample = torch.cat((G_sample, g_sample.detach().cpu()), dim=0)
            cur = cur + opt.batchsize
        else:
            temp_x = test_sim_feature[cur:]
            temp_att = test_sim_att_2[cur:].cuda()
            g_sample, _ = netG(temp_x.cuda(), temp_att)
            G_sample = torch.cat((G_sample, g_sample.detach().cpu()), dim=0)
            cur = cur + opt.batchsize
            break
    
    # for feats, att in zip(test_sim_feature, test_sim_att_2):
    #     g_sample = netG(feats.unsqueeze(dim=0).cuda(), att.unsqueeze(dim=0).cuda())
    #     G_sample = torch.cat((G_sample, g_sample.detach().cpu()), dim=0)

    cls_label = map_label(test_sim_label, dataset.unseenclasses)
    # G_unseen_sample, _ = netG(test_sim_feature.cuda(), test_sim_att_2.view(-1, opt.att_dim).cuda())
    # G_sample = G_unseen_sample.detach().cpu()
    
    # unseen_label = []
    # for i in dataset.test_unseen_label.numpy():
    #     unseen_label.append(i+dataset.ntrain_class)
    # unseen_label = np.array(unseen_label)

    # classifier = SVM(G_sample, test_sim_label)
    # classifier = Softmax(G_sample, test_sim_label)
    # acc, acc2 = classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)
    
    classifier = Classifier(G_sample, cls_label, _nclass = dataset.ntest_class)
    acc, acc2 = classifier.acc(dataset.test_unseen_feature, map_label(dataset.test_unseen_label, dataset.unseenclasses))
    
    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    best = max(acc, acc2)
    if best > result.best_acc:
        result.best_acc = best
        result.best_iter = it
        result.save_model = True
    print("T1 Acc {:.2f}%\tAcc2: {:.2f}%\tBest acc: {:.2f}%".format(acc, acc2, result.best_acc))


def eval_fakefeat_test_gzsl(it, netG, preinputs, opt, result, dataset):
    # test_unseen
    test_sim_feature, test_sim_att_2, test_sim_label = preinputs.test_sim_feature, preinputs.test_sim_att_2.view(-1, opt.att_dim), preinputs.test_sim_label.view(-1)
    netG.eval()
    # G_sample = np.array([]).reshape(-1, opt.x_dim)
    G_sample = torch.randn(0).reshape(-1, opt.x_dim)
    cur = 0
    while(True):
        if cur+opt.batchsize <= len(test_sim_feature):
            temp_x = test_sim_feature[cur:cur+opt.batchsize]
            temp_att = test_sim_att_2[cur:cur+opt.batchsize].cuda()
            g_sample, _ = netG(temp_x.cuda(), temp_att)
            G_sample = torch.cat((G_sample, g_sample.detach().cpu()), dim=0)
            cur = cur + opt.batchsize
        else:
            temp_x = test_sim_feature[cur:]
            temp_att = test_sim_att_2[cur:].cuda()
            g_sample, _ = netG(temp_x.cuda(), temp_att)
            G_sample = torch.cat((G_sample, g_sample.detach().cpu()), dim=0)
            cur = cur + opt.batchsize
            break

    # G_sample = torch.cat((dataset.train_feature, G_sample))
    
    # slabel = dataset.train_label.reshape(-1).long()
    # ulabel = test_sim_label.reshape(-1).long()  # 生成的unseen类的label
    # G_label = torch.cat((slabel, ulabel))
    
    # test_seen
    test_seen_sim_feature, test_seen_sim_att_2, test_seen_sim_label = preinputs.test_seen_sim_feature, preinputs.test_seen_sim_att_2, preinputs.test_seen_sim_label.view(-1)
    cur = 0
    while(True):
        if cur+opt.batchsize <= len(test_seen_sim_feature):
            temp_x = test_seen_sim_feature[cur:cur+opt.batchsize]
            temp_att = test_seen_sim_att_2[cur:cur+opt.batchsize].cuda()
            g_sample, _ = netG(temp_x.cuda(), temp_att)
            G_sample = torch.cat((G_sample, g_sample.detach().cpu()), dim=0)
            cur = cur + opt.batchsize
        else:
            temp_x = test_seen_sim_feature[cur:]
            temp_att = test_seen_sim_att_2[cur:].cuda()
            g_sample, _ = netG(temp_x.cuda(), temp_att)
            G_sample = torch.cat((G_sample, g_sample.detach().cpu()), dim=0)
            cur = cur + opt.batchsize
            break

    slabel = test_seen_sim_label.reshape(-1,1)
    ulabel = test_sim_label.reshape(-1,1)  # 生成的unseen类的label
    G_label = torch.cat((ulabel, slabel)).squeeze()
    
    # unseen_label = []
    # for i in dataset.test_unseen_label.numpy():
    #     unseen_label.append(i+dataset.ntrain_class)
    classifier = Classifier(G_sample, G_label, _nclass = dataset.ntrain_class+dataset.ntest_class)
    
    # classifier = Softmax(G_sample.numpy(), G_label.numpy())
    
    # S-->T
    acc_S_T, acc_S_T2 = classifier.acc(dataset.test_seen_feature, dataset.test_seen_label) # tensor, ndarray

    # U-->T
    acc_U_T, acc_U_T2 = classifier.acc(dataset.test_unseen_feature, dataset.test_unseen_label)

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
        result.best_acc_S_T_std = acc_S_T2
        result.best_acc_U_T_std = acc_U_T2
        
        result.best_iter = it
        result.save_model = True

    out_log = "Metric1.\tH {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  Best H: {:.2f} \n".format(acc, acc_S_T, acc_U_T, result.best_acc)
    out_log += "Metric2.\tH {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  Best H: {:.2f} \n".format(acc2, acc_S_T2, acc_U_T2, result.best_acc_std)
    print(out_log)
    # out_file = f'data/{opt.dataset}/gan/Kcla{opt.k_class}_Kinst{opt.k_inst}_X_w:{opt.x_w}_Centroid_w:{opt.cent_w}_Selected_ratio:{opt.selected_ratio}_Split_num:{opt.split_num}_gzsl.log'
    # with open (out_file, 'a') as out:
    #     out.write(out_log)
    #     print(out_log)
    # classifier = Softmax(G_sample, G_label)
  
class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_acc_std = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.best_acc_S_T_std = 0.0
        self.best_acc_U_T_std = 0.0
        self.acc_list = []
        self.iter_list = []

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.GP_LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    train()

