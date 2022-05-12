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

from dataset import FeatDataLayer, DATA_LOADER, KnnFeat, SVM, Softmax, Classifier
from models import _netD, _netCC_3, _param, Encoder, Generator, _netD_SC

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

parser.add_argument('--z_dim',  type=int, default=300, help='dimension of the intermedia vector z')
parser.add_argument('--disp_interval', type=int, default=40)
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
parser.add_argument('--lr_g', type = float, default = 0.0005, help= 'learning rate of generator')
parser.add_argument('--lr_d', type = float, default = 0.0001, help= 'learning rate of discriminator')
parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value for disc. weights")
parser.add_argument('--cent_w', type = float, default = 10, help= 'CENT_LADMBDA')
parser.add_argument('--extended_attr_num', type = int, default = 0)
parser.add_argument('--vae', type = float, default = 2)

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

    # cprint(" The output dictionary is {}".format(out_subdir), 'red')
    # log_dir = out_subdir + '/log.txt'
    # with open(log_dir, 'w') as f:
    #     f.write('Training Start:')
    #     f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    #     for k, v in zip(vars(opt).keys(),vars(opt).values()):
    #         f.write(k+':'+str(v))

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
    # netG = _netCC_3(preinputs.train_sim_att_2.numpy().shape[1]).cuda()
    # netG = torch.nn.DataParallel(netG)
    visual_dim = dataset.train_feature.numpy().shape[1]
    semantic_dim = preinputs.train_sim_att_2.numpy().shape[1]
    netE = Encoder(_in_dim = visual_dim, _out_dim = semantic_dim).cuda()
    netG = Generator(_in_dim = semantic_dim, _out_dim = visual_dim).cuda()
    netE.apply(weights_init)
    netG.apply(weights_init)
    
    #netD = _netD(dataset.train_cls_num, dataset.feature_dim).cuda()
    netD = _netD_SC(dataset.train_cls_num, dataset.feature_dim).cuda()
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
    nets = [netE, netG, netD]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d)
    optimizerE = optim.Adam(netE.parameters(), lr=opt.lr_g)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g)
    
    # optimizerD = optim.SGD(netD.parameters(), lr=opt.lr_d, betas=(0.5, 0.9))
    # optimizerG = optim.SGD(netG.parameters(), lr=opt.lr_g, betas=(0.5, 0.9))
    # optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr_d, alpha=0.99, eps=1e-08)
    # optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr_g, alpha=0.99, eps=1e-08)      

    D_criterion = nn.NLLLoss().cuda()
    best_acc = 0.0
    for it in range(start_step, opt.endEpoch):
        #print('epoch: ',it)
        ##############################################
        #########Discriminator training ##############
        ##############################################
        gp_sum = 0 #lAMBDA VARIABLE
        for _ in range(opt.D_epoch):
            blobs = data_layer.forward()
            feat_data = blobs['data'] # target class sample
            x = blobs['x'] # source class sample
            att1 = blobs['att1']  # source class sample
            att = blobs['att2'] # target class attribute
            labels = blobs['labels'].astype(int)  # true class labels

            x = Variable(torch.from_numpy(x.astype('float32'))).cuda()
            att1 = Variable(torch.from_numpy(att1.astype('float32'))).cuda()
            att = Variable(torch.from_numpy(att.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data.astype('float32'))).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            # z = Variable(torch.randn(opt.batchsize, opt.z_dim)).cuda()  # random noise

            # GAN's D loss 
            D_real, C_real = netD(X) # D_real: 1 dim score, C_real: predict class
            D_loss_real = torch.mean(D_real) 
            C_loss_real = D_criterion(C_real, y_true)
            DC_loss = opt.Adv_LAMBDA *(-D_loss_real + C_loss_real)
            DC_loss.backward()
            
            # Encoder
            means, log_var = netE(x, att)  
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batchsize, att1.shape[1]]).cuda()
            z = eps * std + means #torch.Size([1024, 85])
            
            # Decoder/Generator
            G_sample = netG(z, c=att)    
              
            D_fake, C_fake = netD(G_sample.detach())
            D_loss_fake = torch.mean(D_fake)
            C_loss_fake = D_criterion(C_fake, y_true)
            DC_loss = opt.Adv_LAMBDA *(D_loss_fake + C_loss_fake)
            DC_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = opt.Adv_LAMBDA * calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward()

            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()
            
            # Clip weights of discriminator
            for p in netD.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
                
            reset_grad(nets)
            
        # gp_sum /= (opt.Adv_LAMBDA*opt.GP_LAMBDA*opt.D_epoch)
        # if (gp_sum > 1.05).sum() > 0:
        #     opt.GP_LAMBDA *= 1.1
        # elif (gp_sum < 1.001).sum() > 0:
        #     opt.GP_LAMBDA /= 1.1

        ##############################################
        ####Encoder+Decoder/Generator training #######
        ##############################################
        for _ in range(opt.G_epoch):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            x = blobs['x'] # 
            att = blobs['att2']
            att1 = blobs['att1']  # source class sample
            labels = blobs['labels'].astype(int)  # class labels

            x = Variable(torch.from_numpy(x.astype('float32'))).cuda()
            att = Variable(torch.from_numpy(att.astype('float32'))).cuda()
            att1 = Variable(torch.from_numpy(att1.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data.astype('float32'))).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()

            # Encoder
            # means, log_var = netE(x, att1)  # 原来的sample+att
            means, log_var = netE(x, att)  # 原来的sample+att
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batchsize, att1.shape[1]]).cuda()
            z = eps * std + means #torch.Size([1024, 85])
            
            # Decoder/Generator
            G_sample = netG(z, c=att)  # 需要生成的类的att     
            
            # VAE loss
            vae_loss_seen = opt.vae * loss_fn(G_sample, X, means, log_var) # 计算生成的样本与真实样本的MSE + VAE中间层的KLD (潜在变量的分布和单位高斯分布的差异)
            
            D_fake, C_fake = netD(G_sample)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # Auxiliary classification loss
            C_loss = D_criterion(C_fake, y_true)
            
            GC_loss = opt.Adv_LAMBDA *(-G_loss + C_loss)

            # GAN's att loss
            # Att_loss = opt.att_w * F.mse_loss(G_att, att)

            # GAN's x loss
            # X_loss = opt.x_w * F.mse_loss(G_sample, X)

            # # Centroid loss
            Euclidean_loss_target = Variable(torch.Tensor([0.0])).cuda()

            if opt.REG_W_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
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
            

            # all_loss = GC_loss + reg_loss + Att_loss + X_loss + Euclidean_loss_target
            all_loss = GC_loss + reg_loss + vae_loss_seen + Euclidean_loss_target
            
            all_loss.backward()
            optimizerE.step()
            optimizerG.step()

            reset_grad(nets)


        if it % opt.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            
            log_text = 'Iter-{}; Was_D:{:.3f}; Euclidean_loss_target:{:.3f};reg_loss:{:.3f}; G_loss: {:.3f}; C_loss:{:.3f}; D_loss_real: {:.3f}; D_loss_fake: {:.3f};  vae_loss_seen:{:.3f}; rl: {:.2f}%; fk: {:.2f}%; '\
                        .format(it, Wasserstein_D.item(),  Euclidean_loss_target.item(),reg_loss.item(),
                                G_loss.item(), C_loss.item(), D_loss_real.item(), D_loss_fake.item(), vae_loss_seen.item(), acc_real * 100, acc_fake * 100)
            print(log_text)
            # with open(log_dir, 'a') as f:
            #     f.write(log_text+'\n')

        if it % opt.evl_interval == 0 and it >= 40:
            netG.eval()
            eval_fakefeat_test(it, netE, netG, preinputs, opt, result, dataset)
            if result.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model_ZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                best_acc = result.acc_list[-1]
                # save_model(it, netG, netD, opt.manualSeed, log_text,
                #            out_subdir + '/Best_model_ZSL_Acc_{:.2f}_Epoch_{:d}.tar'.format(result.acc_list[-1],it))
            if opt.gzsl:
                eval_fakefeat_test_gzsl(it, netE, netG, preinputs, opt, result_gzsl, dataset)
                if result_gzsl.save_model:
                    files2remove = glob.glob(out_subdir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    # best_acc_gzsl = result.acc_list[-1]
                    # save_model(it, netG, netD, opt.manualSeed, log_text,
                    #            out_subdir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}_Epoch_{:d}.tar'.format(result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T,it))

            netG.train()
    result_file = 'gzsl_results/gan_result_'+opt.dataset+'_vae.txt'
    with open(result_file,'a') as out:
        out_info = 'Kcla{:d}_Kinst{:d}_Att_w:{}_X_w:{}_Cent_W:{}_VAE_W:{}_Acc:{:.3f}\n'.format(opt.k_class,opt.k_inst,opt.att_w, opt.x_w, opt.cent_w, opt.vae, best_acc)
        out_info += 'GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}\n'.format(result_gzsl.best_acc, result_gzsl.best_acc_S_T, result_gzsl.best_acc_U_T)
        out.write(out_info)
        
        

def save_model(it, netG, netD, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'state_dict_D': netD.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)

    
def generate_syn_feature(generator, classes, attribute, syn_num=500, resSize=2048):
    nclass = classes.size(0)
    att_dim = attribute.shape[1]
    syn_feature = torch.FloatTensor(nclass*syn_num, resSize)
    syn_label = torch.LongTensor(nclass*syn_num) 
    syn_att = torch.FloatTensor(syn_num, att_dim).cuda()
    syn_noise = torch.FloatTensor(syn_num, att_dim).cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(syn_num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise,volatile=True)
        syn_attv = Variable(syn_att,volatile=True)
        fake = generator(syn_noisev,c=syn_attv)
        # if netF is not None:
        #     dec_out = netDec(fake) # only to call the forward function of decoder
        #     dec_hidden_feat = netDec.getLayersOutDet() #no detach layers
        #     feedback_out = netF(dec_hidden_feat)
        #     fake = generator(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*syn_num, syn_num).copy_(output.data.cpu())
        syn_label.narrow(0, i*syn_num, syn_num).fill_(i)

    return syn_feature.detach().cpu().numpy(), syn_label.detach().cpu().numpy()

def eval_fakefeat_test(it, netE, netG, preinputs, opt, result, dataset):
    test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label = preinputs.test_sim_feature.numpy(), preinputs.test_sim_att_1.numpy(), preinputs.test_sim_att_2.numpy(), preinputs.test_sim_label.numpy()
    G_sample = np.zeros([0, 2048])
    
    temp_x = Variable(torch.from_numpy(test_sim_feature.astype('float32'))).cuda()
    temp_att1 = Variable(torch.from_numpy(test_sim_att_1.astype('float32'))).cuda()
    temp_att = Variable(torch.from_numpy(test_sim_att_2.astype('float32'))).cuda()
    
    means, log_var = netE(temp_x, temp_att1)  # 原来的sample+att
    std = torch.exp(0.5 * log_var)
    eps = torch.randn([temp_att1.shape[0], temp_att1.shape[1]]).cuda()
    z = eps * std + means #torch.Size([1024, 85])
    G_sample = netG(z, c=temp_att).detach().cpu().numpy()  # 需要生成的类的att     
            
    # unseen_label = []
    # for i in dataset.test_unseen_label.numpy():
    #     unseen_label.append(i+dataset.ntrain_class)
    # unseen_label = np.array(unseen_label)
    # G_sample, G_test_label = generate_syn_feature(netG, dataset.unseenclasses, dataset.attribute, syn_num=500)
    # classifier = SVM(G_sample, test_sim_label)
    # classifier = Softmax(G_sample, test_sim_label)
    # acc, acc2 = classifier.acc(dataset.test_unseen_feature.numpy(), unseen_label)
    classifier = Classifier(G_sample, test_sim_label-dataset.ntrain_class, _nclass = dataset.ntest_class)
    acc, acc2 = classifier.acc(dataset.test_unseen_feature, dataset.test_unseen_label.numpy())
    
    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True
    print("Accuracy is {:.2f}%, standard acc : {:.2f}%, Best acc: {:.2f}%".format(acc, acc2, result.best_acc))


def eval_fakefeat_test_gzsl(it, netE, netG, preinputs, opt, result, dataset):
    test_sim_feature, test_sim_att_1, test_sim_att_2, test_sim_label = preinputs.test_sim_feature.numpy(), preinputs.test_sim_att_1.numpy(), preinputs.test_sim_att_2.numpy(), preinputs.test_sim_label.numpy()
    
    # test_unseen_sample
    temp_x = Variable(torch.from_numpy(test_sim_feature.astype('float32'))).cuda()
    temp_att1 = Variable(torch.from_numpy(test_sim_att_1.astype('float32'))).cuda()
    temp_att = Variable(torch.from_numpy(test_sim_att_2.astype('float32'))).cuda()
    
    means, log_var = netE(temp_x, temp_att1)  # 原来的sample+att
    std = torch.exp(0.5 * log_var)
    eps = torch.randn([temp_att1.shape[0], temp_att1.shape[1]]).cuda()
    z = eps * std + means #torch.Size([1024, 85])
    G_unseen_sample = netG(z, c=temp_att).detach().cpu().numpy()  # 需要生成的类的att   
    # test_seen
    test_seen_sim_feature, test_seen_sim_att_2, test_seen_sim_label = preinputs.test_seen_sim_feature.numpy(), preinputs.test_seen_sim_att_2.numpy(), preinputs.test_seen_sim_label.numpy()
    G_seen_sample = test_seen_sim_feature
    
    G_sample = np.vstack((G_seen_sample, G_unseen_sample))

    slabel = test_seen_sim_label.reshape(-1,1)
    ulabel = test_sim_label.reshape(-1,1)  # 生成的unseen类的label
    G_label = np.vstack((slabel, ulabel)).squeeze()

    unseen_label = []  # 用于测试的unseen类的label
    for i in dataset.test_unseen_label.numpy():
        unseen_label.append(i+dataset.ntrain_class)
    unseen_label = np.array(unseen_label)

    classifier = Classifier(G_sample, G_label, _nclass = dataset.ntrain_class+dataset.ntest_class)
    
    # S-->T
    acc_S_T, acc_S_T2 = classifier.acc(dataset.test_seen_feature, dataset.test_seen_label.numpy()) # tensor, ndarray

    # U-->T
    acc_U_T, acc_U_T2 = classifier.acc(dataset.test_unseen_feature, unseen_label)

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

    out_log = "H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  \nStdAcc: {:.2f}\tBest Acc: {:.2f}\t Best Acc_Std:{:.2f}\n".format(acc, acc_S_T, acc_U_T, acc2, result.best_acc, result.best_acc_std)
    out_file = f'data/AwA2/Kcla{opt.k_class}_Kinst{opt.k_inst}_Att_w:{opt.att_w}_X_w:{opt.x_w}_Centroid_w:{opt.cent_w}_VAE:{opt.vae}_gzsl.txt'
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

def loss_fn(G_x, x, mean, log_var):
    MSE = F.mse_loss(G_x, x.detach())
    MSE = MSE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (MSE + KLD)

if __name__ == "__main__":
    train()

