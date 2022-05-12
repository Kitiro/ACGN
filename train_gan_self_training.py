# coding:utf8
# 毕设将bmvc的cross_gan改为对抗式和非对抗式
# 本脚本为对抗式
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
from scipy import io as sio

from dataset import FeatDataLayer, DATA_LOADER, KnnFeat, SVM, Softmax
from models import _netD, _netCC_3, _param

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

parser.add_argument('--z_dim',  type=int, default=300, help='dimension of the random vector z')
parser.add_argument('--disp_interval', type=int, default=40)
parser.add_argument('--save_interval', type=int, default=300)
parser.add_argument('--evl_interval',  type=int, default=20)

parser.add_argument('--random', action='store_true', default=False, help = 'random pairs in data preparation')
parser.add_argument('--endEpoch', type = int, default = 2000, help= 'train epoch')
parser.add_argument('--gzsl', action='store_true', default = False, help = 'gzsl evaluation')
parser.add_argument('--batchsize', type = int, default = 1024, help= 'batchsize')
parser.add_argument('--k_class', type = int, default = 1, help= 'find k similar classes')
parser.add_argument('--k_inst', type = int, default = 1, help= 'find k similar instances in each similar class')
parser.add_argument('--att_w', type = int, default = 10, help= 'weight of Att_loss')
parser.add_argument('--x_w', type = int, default = 40, help= 'weight of X_loss')
parser.add_argument('--lr_g', type = float, default = 1e-5, help= 'learning rate of generator')
parser.add_argument('--lr_d', type = float, default = 5e-5, help= 'learning rate of discriminator')
parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value for disc. weights")
parser.add_argument('--cent_w', type = float, default = 10, help= 'CENT_LADMBDA')
parser.add_argument('--extended_attr_num', type = int, default = 0)
parser.add_argument('--s_w', type = float, default = 1, help= 'weight of Semantic_loss for Seen class')
parser.add_argument('--u_w', type = float, default = 1, help= 'weight of Semantic_loss for Unseen class')

opt = parser.parse_args()
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.GP_LAMBDA = 10    # Gradient penalty lambda
opt.CENT_LAMBDA  = opt.cent_w
opt.REG_W_LAMBDA = 0.001
opt.Adv_LAMBDA = 1

#opt.lr = 0.0001

""" hyper-parameter for testing"""
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20      # knn: the value of K
opt.in_features = 600
opt.out_features = 2048
opt.G_epoch = 1
opt.D_epoch = 2
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

    out_dir  = opt.path_root + '/gan'
    if not opt.random:
        out_subdir = opt.path_root + '/gan/Kcla{:d}_Kinst{:d}_Att_w:{}_X_w:{}_Centroid_w:{}_{:s}'.format(opt.k_class,opt.k_inst,opt.att_w, opt.x_w, opt.cent_w, timestamp)
        # out_subdir = opt.path_root + '/gan/Kcla{:d}_Kinst{:d}_{:s}'.format(opt.k_class,opt.k_inst,timestamp)
    else:
        out_subdir = opt.path_root + '/gan/Random_{:s}'.format(timestamp)

    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # if not os.path.exists(out_subdir):
    #     os.mkdir(out_subdir)

    cprint("The output dictionary is {}".format(out_subdir), 'red')
    log_dir = out_subdir + '_log.txt'
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
        for k, v in zip(vars(opt).keys(),vars(opt).values()):
            f.write(k+':'+str(v))

    # Tensor = torch.cuda.FloatTensor
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
    # netG = torch.nn.DataParallel(netG)
    netG.apply(weights_init)
    
    # netD = _netD(dataset.train_cls_num, dataset.feature_dim).cuda()
    netD = _netD(dataset.train_cls_num + dataset.test_cls_num, dataset.feature_dim).cuda()

    # netD = torch.nn.DataParallel(netD)
    netD.apply(weights_init)

    class_dic = {} # 记录每一个训练类在train_classes中的最相似类的序号
    class_knn = NearestNeighbors(n_neighbors = opt.k_class+1, metric = 'cosine').fit(dataset.train_att)
    for i in np.unique(dataset.train_label.numpy()):
        d, ind = class_knn.kneighbors(dataset.train_att[i].reshape((1,-1)))
        class_dic[i] = ind[0][1:]

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

    # 映射后的semantic feature
    mapping_semantic = torch.from_numpy(sio.loadmat('data/AwA2/mapped_semantic.mat')['mapped_feature'])
    #mapping_semantic_seen = mapping_semantic[dataset.seenclasses]
    #print(preinputs.test_sim_label)
    #print(preinputs.test_sim_label.shape)
    mapping_semantic_unseen = mapping_semantic[preinputs.test_sim_label.long()].cuda() # 产生的unseen类的label对应的mapped semantic feature
    # print('mapping feature shape:', mapping_semantic_unseen.shape)
    
    generated_unseen_x = preinputs.test_sim_feature.cuda()
    generated_unseen_att = preinputs.test_sim_att_2.cuda()
    generated_unseen_label = preinputs.test_sim_label.long().cuda() # 40 - 49

    best_acc = 0.0
    for it in range(start_step, opt.endEpoch):
        #print('epoch: ',it)
        """ Discriminator """
        while True:
            blobs = data_layer.forward()
            feat_data = blobs['data'] # target class sample
            x = blobs['x'] # source class sample
            att1 = blobs['att1']
            att = blobs['att2'] # target class attribute
            labels = blobs['labels'].astype(int)  # true class labels

            x = Variable(torch.from_numpy(x.astype('float32'))).cuda()
            att = Variable(torch.from_numpy(att.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data.astype('float32'))).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, opt.z_dim)).cuda()  # random noise

            # GAN's D loss 
            D_real, C_real = netD(X) # D_real: 1 dim score, C_real: predict class
            D_loss_real = torch.mean(D_real) 
            C_loss_real = F.cross_entropy(C_real, y_true)
            DC_loss = opt.Adv_LAMBDA *(-D_loss_real + C_loss_real)
            DC_loss.backward()
            
            # seen class
            G_sample_seen, _ = netG(x, att) # generated sample, hidden state
            G_sample_seen = G_sample_seen.detach()
            D_fake_seen, C_fake_seen = netD(G_sample_seen)
            D_loss_fake_seen = torch.mean(D_fake_seen)

            # unseen的semantic feature经映射后，作为unseen的ground truth
            G_sample_unseen, _ = netG(generated_unseen_x, generated_unseen_att)
            G_sample_unseen = G_sample_unseen.detach()
            D_fake_unseen, C_fake_unseen = netD(G_sample_unseen)
            D_loss_fake_unseen = torch.mean(D_fake_unseen)
            
            C_loss_fake_seen = F.cross_entropy(C_fake_seen, y_true)  # 分类损失
            C_loss_fake_unseen = F.cross_entropy(C_fake_unseen, generated_unseen_label)  # 分类损失
            DC_loss_seen = opt.Adv_LAMBDA *(D_loss_fake_seen + C_loss_fake_seen)
            DC_loss_seen.backward()
            DC_loss_unseen = opt.Adv_LAMBDA *(D_loss_fake_unseen + C_loss_fake_unseen)
            DC_loss_unseen.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = opt.Adv_LAMBDA * calc_gradient_penalty(netD, X.data, G_sample_seen.data)
            grad_penalty.backward()

            Wasserstein_D = D_loss_real - D_loss_fake_seen - D_loss_fake_unseen # need increment
            optimizerD.step()
            
            # Clip weights of discriminator
            for p in netD.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
                
            reset_grad(nets)
            if blobs['newEpoch']:
                break


        """ Generator """
        while True:
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            x = blobs['x'] # 
            att = blobs['att2']
            labels = blobs['labels'].astype(int)  # class labels

            x = Variable(torch.from_numpy(x.astype('float32'))).cuda()
            att = Variable(torch.from_numpy(att.astype('float32'))).cuda()

            # valid = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)

            X = Variable(torch.from_numpy(feat_data.astype('float32'))).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()

            # 产生seen类 & 产生unseen类
            G_sample_seen, G_att_seen = netG(x, att)
            G_sample_unseen, G_att_unseen = netG(generated_unseen_x, generated_unseen_att)

            D_fake_seen, C_fake_seen = netD(G_sample_seen) # G_sample:generated seen sample
            D_fake_unseen, C_fake_unseen = netD(G_sample_unseen) # G_sample:generated unseen sample
            D_real,      C_real = netD(X) # X:real sample 

            # GAN's D loss
            G_loss = torch.mean(D_fake_seen) + torch.mean(D_fake_unseen)
            
            # Auxiliary classification loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake_seen, y_true) + F.cross_entropy(C_fake_unseen, generated_unseen_label)) / 3
            GC_loss = opt.Adv_LAMBDA *(-torch.mean(D_real) + G_loss + C_loss)

            # GAN's att loss
            Att_loss = opt.att_w * F.mse_loss(G_att_seen, att) 
            Att_loss_unseen = opt.att_w * F.mse_loss(G_att_unseen, generated_unseen_att) 

            # GAN's x loss
            X_loss = opt.x_w * F.mse_loss(G_sample_seen, X)  # 生成样本与真实样本进行MSE
            X_loss_unseen = opt.x_w*0.5 * F.mse_loss(G_sample_unseen, mapping_semantic_unseen) # 生成unseen样本，与unseen的attributes经映射到visual space后的特征进行MSE

            # # Centroid loss
            Euclidean_loss_target = Variable(torch.Tensor([0.0])).cuda()

            if opt.REG_W_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss_target += 0.0

                    else:
                        G_sample_cls = G_sample_seen[sample_idx, :]
                        Euclidean_loss_target += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
                        
                Euclidean_loss_target *= 1.0/dataset.train_cls_num * opt.CENT_LAMBDA
                

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)
            

            all_loss = GC_loss + reg_loss + Att_loss + X_loss + Euclidean_loss_target + X_loss_unseen + Att_loss_unseen
            all_loss.backward()

            optimizerG.step()

            reset_grad(nets)
            if blobs['newEpoch']:
                break


        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake_seen.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake_unseen = (np.argmax(C_fake_unseen.data.cpu().numpy(), axis=1) == generated_unseen_label.data.cpu().numpy()).sum() / float(generated_unseen_label.data.size()[0])
            
            log_text = 'Iter-{}; Was_D:{:.3f}; Euclidean_loss_target:{:.3f};reg_loss:{:.3f}; G_loss: {:.3f}; D_loss_real: {:.3f}; D_loss_fake_seen: {:.3f}; D_loss_fake_unseen: {:.3f}; att_loss:{:.3f}; x_loss:{:.3f} ;x_loss_unseen:{:.3f}; rl: {:.2f}%; fk: {:.2f}%; fk_unseen: {:.2f}%; '\
                        .format(it, Wasserstein_D.item(),  Euclidean_loss_target.item(),reg_loss.item(),
                                G_loss.item(), D_loss_real.item(), D_loss_fake_seen.item(), D_loss_fake_unseen.item(), Att_loss.item(), X_loss.item(), X_loss_unseen.item(), acc_real * 100, acc_fake * 100, acc_fake_unseen * 100)
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
                best_acc = result.acc_list[-1]
                # save_model(it, netG, netD, opt.manualSeed, log_text,
                #            out_subdir + '/Best_model_ZSL_Acc_{:.2f}_Epoch_{:d}.tar'.format(result.acc_list[-1],it))
            if opt.gzsl:
                eval_fakefeat_test_gzsl(it, netG, preinputs, opt, result_gzsl, dataset)
                if result_gzsl.save_model:
                    files2remove = glob.glob(out_subdir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    # best_acc_gzsl = result.acc_list[-1]
                    # save_model(it, netG, netD, opt.manualSeed, log_text,
                    #            out_subdir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}_Epoch_{:d}.tar'.format(result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T,it))

            netG.train()
    # result_file = 'gzsl_results/gan_result_'+opt.dataset+'.txt'
    # with open(result_file,'a') as out:
    #     out_info = 'Kcla{:d}_Kinst{:d}_Att_w:{}_x_w:{}_Std:{}_Acc{}\n'.format(opt.k_class,opt.k_inst,opt.att_w, opt.x_w, str(opt.standardization), best_acc)
    #     # out_info += 'GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}\n'.format(result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T)
    #     out.write(out_info)
    with open(f'data/AwA2/gan/Kcla{opt.k_class}_Kinst{opt.k_inst}_self_traing_results.txt', 'a') as out:
        out.write('Att_w:{}_X_w:{}_Centroid_w:{}_Semantic_w:{}\nAcc:{}_StdAcc:{}'.format(opt.att_w, opt.x_w, opt.cent_w, opt.s_w, result.best_acc, result.best_acc_std))
    
        

def save_model(it, netG, netD, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'state_dict_D': netD.state_dict(),
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
    acc, acc2 = classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True
    print("Accuracy is {:.2f}%, standard acc : {:.2f}%".format(acc, acc2))


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

