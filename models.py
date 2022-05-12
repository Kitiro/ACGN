# coding:utf8
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import math
import numpy as np

rdc_text_dim = 1000
z_dim = 100
h_dim = 1024

class _param:
    def __init__(self):
        self.rdc_text_dim = rdc_text_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.text_dim = 300


class _netD(nn.Module):
    """[summary]
      D_gan(h): 1-dim score
      D_aux(h): classification loss
    """

    def __init__(self, y_dim=40, X_dim=2048):
        super(_netD, self).__init__()
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, h_dim), nn.ReLU())
        # Discriminator net branch one: For Gan_loss
        self.D_gan = nn.Sequential(nn.Linear(h_dim, 1), nn.ReLU())
        # self.D_gan = nn.Linear(h_dim, 1)

        # Discriminator net branch two: For aux cls loss
        self.D_aux = nn.Linear(h_dim, y_dim)
        # self.D_aux = nn.Sequential(nn.Linear(h_dim, y_dim), nn.ReLU())
        # for p in self.parameters(): p.requiresgrad=False

    def forward(self, input):
        h = self.D_shared(input)
        return self.D_gan(h), self.D_aux(h)


class _netD_att(nn.Module):
    # discriminator for attribute
    def __init__(self, y_dim=40, X_dim=2048, A_dim=85):
        super(_netD_att, self).__init__()

        # Discriminator net layer one
        self.DA_shared = nn.Sequential(nn.Linear(X_dim, A_dim), nn.ReLU())
        self.DA_gan = nn.Sequential(nn.Linear(A_dim, 1), nn.ReLU())
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, h_dim), nn.ReLU())
        self.D_gan = nn.Sequential(nn.Linear(h_dim, 1), nn.ReLU())
        self.D_aux = nn.Linear(h_dim, y_dim)

    def forward(self, input):
        h = self.D_shared(input)
        a = self.DA_shared(input)
        return self.D_gan(h), self.D_aux(h), self.DA_gan(a)


class _netD_SC(nn.Module):
    """ using as the netD in GAN+SC
    """
    def __init__(self, y_dim=50, X_dim=2048):
        super(_netD_SC, self).__init__()
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, h_dim), nn.ReLU())
        # Discriminator net branch one: For Gan_loss
        self.D_gan = nn.Sequential(nn.Linear(h_dim, 1))
        # Discriminator net branch two: For classification loss
        self.D_cl = nn.Sequential(
            nn.Linear(h_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, y_dim),
            nn.ReLU(),
        )

    def forward(self, input):
        h = self.D_shared(input)
        return self.D_gan(h), self.D_cl(h)  # TODO label smooth

    
class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, A, X):
        return A.mm(X).mm(self.weight)

class Adj(nn.Module):
    """ 计算新的关系矩阵
    """

    def __init__(self, t=0.01):
        super(Adj, self).__init__()
        self.t = t

    def _adj(self, X):
        feat = X.data.cpu().numpy()
        l = len(feat)
        adj = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                adj[i][j] = np.e ** (-np.linalg.norm(feat[i] - feat[j]) / self.t)
        return torch.from_numpy(adj).float().cuda()

    def forward(self, X):
        return self._adj(X)


class _netGCN_SC(nn.Module):
    """Using semantic + noise to build a classifier. 
    in_dim: input dims
    out_dim: class num
    """

    def __init__(self, in_dim=600, out_dim=50):
        super(_netGCN_SC, self).__init__()
        self.graph1 = GraphConv(in_dim, 1024)
        self.relu = nn.ReLU()
        self.graph2 = GraphConv(1024, 2048)
        self.linear = nn.Linear(2048, out_dim)
        self.softmax = nn.LogSoftmax()

    def forward(self, A, X, Z):
        x = torch.cat([X, Z], 1)
        x = self.relu(self.graph1(A, x))
        feature = self.relu(self.graph2(A, x))
        x = self.linear(feature)
        x = self.softmax(x)
        return feature, x


class _netGCN_G(nn.Module):
    """Using semantic + noise to generate features
    """

    def __init__(self, in_dim=600, out_dim=2048):
        super(_netGCN_G, self).__init__()
        self.graph1 = GraphConv(in_dim, 1024)
        self.relu = nn.ReLU()
        self.graph2 = GraphConv(1024, out_dim)

    def forward(self, A, X, Z):
        x = torch.cat([X, Z], 1)
        x = self.relu(self.graph1(A, x))
        x = self.relu(self.graph2(A, x))
        return x


class _netGCN(nn.Module):
    """Using semantic + noise to generate fake features. 
    in_dim: semantic dim + noise dim
    out_dim: fake feature dim (2048)
    """

    def __init__(self, in_dim, out_dim):
        super(_netGCN, self).__init__()
        self.graph1 = GraphConv(in_dim, 1024)
        self.relu = nn.ReLU()
        self.graph2 = GraphConv(1024, out_dim)

    def forward(self, A, X, Z):
        x = torch.cat([X, Z], 1)
        x = self.relu(self.graph1(A, x))
        x = self.relu(self.graph2(A, x))
        return x


class _netGCN2(nn.Module):
    def __init__(self, in_features, out_features):
        super(_netGCN2, self).__init__()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.graph1 = GraphConv(in_features, 1024)
        self.graph2 = GraphConv(1024, 512)
        self.graph3 = GraphConv(512, 256)
        self.graph4 = GraphConv(256, 512)
        self.graph5 = GraphConv(512, 1024)
        self.graph6 = GraphConv(1024, out_features)
        self.adj = Adj()

    def forward(self, A, X):
        output = self.lrelu(self.graph1(A, X))
        output = self.lrelu(self.graph2(A, output))
        output = self.lrelu(self.graph3(A, output))
        a = self.adj(output)
        output = self.lrelu(self.graph4(A, output))
        output = self.lrelu(self.graph5(A, output))
        output = self.relu(self.graph6(A, output))

        return output, a


class _netCC_1(nn.Module):
    def __init__(self, att_dim, in_dim=2048, out_dim=2048):
        super(_netCC_1, self).__init__()
        self.att_dim = att_dim
        self.down = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.att_dim),
            nn.ReLU(),
        )

        self.up = nn.Sequential(
            nn.Linear(self.att_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.ReLU(),
        )

        self.linear = nn.Linear(2 * self.att_dim, self.att_dim)
        self.relu = nn.ReLU()

    def forward(self, x, att1, att2):

        output = self.down(x)
        hiden1 = torch.cat([output, att1], 1)
        hiden2 = self.linear(hiden1)
        hiden2 = self.relu(hiden2)
        hiden3 = torch.cat([hiden2, att2], 1)
        hiden4 = self.linear(hiden3)
        hiden4 = self.relu(hiden4)
        output = self.up(hiden4)
        return output, hiden2[:, 0 : self.att_dim], hiden4[:, 0 : self.att_dim]


class _netCC_2(nn.Module):
    def __init__(self, att_dim, in_dim=2048, out_dim=2048):
        super(_netCC_2, self).__init__()
        self.att_dim = att_dim
        self.down = nn.Sequential(
            nn.Linear(in_dim + self.att_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.att_dim),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.Linear(self.att_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.ReLU(),
        )

    def forward(self, x, att):
        output = torch.cat([x, att], 1)
        hiden_att = self.down(output)
        output = self.up(hiden_att)

        return output, hiden_att


class _netCC_3(nn.Module):
    def __init__(self, att_dim, in_dim=2048, out_dim=2048):
        super(_netCC_3, self).__init__()
        self.att_dim = att_dim
        self.down1 = nn.Sequential(nn.Linear(in_dim + self.att_dim, 1024), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(1024 + self.att_dim, 512), nn.ReLU())
        self.down3 = nn.Sequential(nn.Linear(512 + self.att_dim, 256), nn.ReLU())
        self.down4 = nn.Sequential(
            nn.Linear(256 + self.att_dim, self.att_dim), nn.ReLU()
        )

        self.up1 = nn.Sequential(nn.Linear(self.att_dim, 256), nn.ReLU())
        self.up2 = nn.Sequential(nn.Linear(256 + self.att_dim, 512), nn.ReLU())
        self.up3 = nn.Sequential(nn.Linear(512 + self.att_dim, 1024), nn.ReLU())
        self.up4 = nn.Sequential(nn.Linear(1024 + self.att_dim, out_dim), nn.ReLU())

    def forward(self, feature1, att2):
        x = torch.cat([feature1, att2], 1)  # concate dimension = 1， column-wisely
        x = self.down1(x)
        x = torch.cat([x, att2], 1)
        x = self.down2(x)
        x = torch.cat([x, att2], 1)
        x = self.down3(x)
        x = torch.cat([x, att2], 1)
        a = self.down4(x)
        x = self.up1(a)
        x = torch.cat([x, att2], 1)
        x = self.up2(x)
        x = torch.cat([x, att2], 1)
        x = self.up3(x)
        x = torch.cat([x, att2], 1)
        x = self.up4(x)

        return x, a

class _netA(nn.Module):
    """ transforming attribute into visual space to obtaining accurate attributes.
    """
    def __init__(self, att_dim=85, v_dim=2048, class_num=50):
        super(_netA, self).__init__()
        # Discriminator net layer one
        self.fc1 = nn.Sequential(nn.Linear(att_dim, att_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(att_dim, v_dim), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(v_dim, class_num))

    def forward(self, input, train=False):
        mid = self.fc1(input)
        vout = self.fc2(mid)
        logits = self.fc3(vout)
        if train:
            return vout, logits
        else:
            return mid

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-2, keepdim=True)
        s = (x - u).pow(2).mean(-2, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
 
# 用整个global features做attention
class _netCC_attention_multi_head(nn.Module):
    def __init__(self, opt):
        super(_netCC_attention_multi_head, self).__init__()
        self.att_dim = opt.att_dim
        self.k_inst = opt.k_inst  # num of samples for generating 1 sample
        self.num_attention_heads = opt.split_num  #  将2048维的visual feature 拆分成多个片段，进行attention
        self.hidden_dim = opt.x_dim

        self.attention_head_size = int(self.hidden_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # query value with attribute 
        self.query = nn.Sequential(nn.Linear(self.att_dim, self.attention_head_size), nn.ReLU())
        self.key = nn.Sequential(nn.Linear(self.hidden_dim, self.all_head_size), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(self.hidden_dim, self.all_head_size), nn.ReLU())

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        # self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)  # 一组内的k个instance
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )       
        self.down1 = nn.Sequential(nn.Linear(self.hidden_dim + self.att_dim, 1024), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(1024 + self.att_dim, 512), nn.ReLU())
        self.down3 = nn.Sequential(nn.Linear(512 + self.att_dim, 256), nn.ReLU())
        self.down4 = nn.Sequential(
            nn.Linear(256 + self.att_dim, self.att_dim), nn.ReLU()
        )

        self.up1 = nn.Sequential(nn.Linear(self.att_dim, 256), nn.ReLU())
        self.up2 = nn.Sequential(nn.Linear(256 + self.att_dim, 512), nn.ReLU())
        self.up3 = nn.Sequential(nn.Linear(512 + self.att_dim, 1024), nn.ReLU())
        self.up4 = nn.Sequential(nn.Linear(1024 + self.att_dim, self.hidden_dim), nn.ReLU())


    def transpose_for_scores(self, x):  # batch, k_inst, split_num 
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # 512, 4, 4, 512
        x = x.view(*new_x_shape) 
        return x.permute(0, 2, 1, 3)   # 512, 4, 4, 512 
    
    def forward(self, source_tensor, target_att, train=False):
        # source_tensor = self.LayerNorm(source_tensor)
        source_tensor = source_tensor.view(-1, self.k_inst, self.hidden_dim)

        mixed_query_layer = self.query(target_att)  # 512, 85
        # mixed_key_layer = self.key(source_tensor)  # 512, 4, 2048
        # mixed_value_layer = self.value(source_tensor) 
        mixed_key_layer = source_tensor  # 512, 4, 2048
        mixed_value_layer = source_tensor

        # 将query复制attention head次，与每一个head上的k_inst个inst相乘
        query_layer = mixed_query_layer.view(-1, 1, 1, self.attention_head_size).repeat_interleave(self.num_attention_heads, dim=1) 
        
        # query_layer = mixed_query_layer.view(-1, self.num_attention_heads, 1, self.attention_head_size)
        
        key_layer = self.transpose_for_scores(mixed_key_layer)  # 512, 4, 2048
        value_layer = self.transpose_for_scores(mixed_value_layer)  

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (512, 8, 1, 256) * (512, 8, 256, 4)  ->  (512, 8, 4) 
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # batchsize, 8, 4, 4 得到四个分数
        
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # attention_probs = self.attn_dropout(attention_probs)
        #  (512, 8, 4) * (512, 8, 4, 256)  -> 
        context_layer = torch.matmul(attention_probs, value_layer) # batchsize, 8, 4, 256 
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # batchsize, 4, 8, 256 
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        new_context_layer_shape = [-1, self.all_head_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # if train:
        #     return self.fc(context_layer), context_layer
        # else:
        #     return self.fc(context_layer)
        
        return self.forward_(context_layer, target_att, train)
    
    def forward_(self, x, att2, train):
        x = torch.cat([x, att2], dim=1)
        x = self.down1(x)
        x = torch.cat([x, att2], 1)
        x = self.down2(x)
        x = torch.cat([x, att2], 1)
        x = self.down3(x)
        x = torch.cat([x, att2], 1)
        a = self.down4(x)
        
        x = self.up1(a)
        x = torch.cat([x, att2], 1)
        x = self.up2(x)
        x = torch.cat([x, att2], 1)
        x = self.up3(x)
        x = torch.cat([x, att2], 1)
        x = self.up4(x)
        if train:
            return x, a
        else:
            return x

# 用整个global features做attention
class _netCC_attention(nn.Module):
    def __init__(self, opt):
        super(_netCC_attention, self).__init__()
        self.att_dim = opt.att_dim
        self.k_inst = opt.k_inst  # num of samples for generating 1 sample
        self.num_attention_heads = opt.split_num  #  将2048维的visual feature 拆分成多个片段，进行attention
        self.hidden_dim = opt.x_dim 
       
        self.query = nn.Linear(opt.att_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)  # 一组内的k个instance
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        

    def forward(self, source_tensor, target_att):
        source_tensor = self.LayerNorm(source_tensor)
        query_layer = self.query(target_att.unsqueeze(dim=1))# batchsize, 1, 2048
        key_layer = self.key(source_tensor)
        value_layer = self.value(source_tensor)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [1, 2048] * [2048, 4]
        attention_scores = attention_scores / math.sqrt(self.hidden_dim)
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer).squeeze()
        # context_layer = torch.mean(context_layer, dim=-2)  # batchsize, 8, 256 . 权重加和
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # batchsize, 4, 8, 256 
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # residual = torch.mean(source_tensor, dim=-2)

        hidden_states = self.fc(context_layer)
        # hidden_states = self.fc(context_layer)
        # hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + source_tensor)
        # hidden_states = torch.mean(hidden_states, dim=-2)
        
        return hidden_states 
    
class _netCC_3_self_attention(nn.Module):
    def __init__(self, opt):
        super(_netCC_3_self_attention, self).__init__()
        self.att_dim = opt.att_dim
        self.k_inst = opt.k_inst  # num of samples for generating 1 sample
        self.num_attention_heads = opt.split_num  #  将2048维的visual feature 拆分成多个片段，进行attention
        self.feat_dim = opt.x_dim 
        self.split_num = opt.split_num  # 将2048维的visual feature 拆分成多个片段，进行attention
        self.projection = nn.Sequential(
            nn.Linear(int(self.feat_dim/self.split_num), 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
        )
       
        self.down1 = nn.Sequential(nn.Linear(self.feat_dim + self.att_dim, 1024), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(1024 + self.att_dim, 512), nn.ReLU())
        self.down3 = nn.Sequential(nn.Linear(512 + self.att_dim, 256), nn.ReLU())
        self.down4 = nn.Sequential(
            nn.Linear(256 + self.att_dim, self.att_dim), nn.ReLU()
        )

        self.up1 = nn.Sequential(nn.Linear(self.att_dim, 256), nn.ReLU())
        self.up2 = nn.Sequential(nn.Linear(256 + self.att_dim, 512), nn.ReLU())
        self.up3 = nn.Sequential(nn.Linear(512 + self.att_dim, 1024), nn.ReLU())
        self.up4 = nn.Sequential(nn.Linear(1024 + self.att_dim, self.feat_dim), nn.ReLU())

    def forward(self, feature1, att2):
            
        x = torch.zeros((feature1.shape[0], self.feat_dim)).float().cuda()  # for input
        split_width = int(self.feat_dim/self.split_num)
        # feature1 = feature1.view((feature1.shape[0], self.k_inst, self.split_num, split_width))  # 将每一批的visual feature 切割成例如4份，一起进行attention。
        feature1 = feature1.view((feature1.shape[0], self.k_inst, self.split_num, split_width)).permute(0, 2, 1, 3)  # 将每一批的visual feature 切割成例如4份，一起进行attention。
        # shape of each feature : [4, 2048] -> [16, 512]
        x = self.attention(feature1).view(feature1.shape[0], self.feat_dim)
        # x = [self.attention(feature).cpu().detach().numpy() for feature in feature1]
        # x = torch.from_numpy(np.array(x)).cuda()
        att2 = att2.float().cuda()
        
        x = torch.cat([x, att2], 1)
        x = self.down1(x)
        x = torch.cat([x, att2], 1)
        x = self.down2(x)
        x = torch.cat([x, att2], 1)
        x = self.down3(x)
        x = torch.cat([x, att2], 1)
        a = self.down4(x)
        x = self.up1(a)
        x = torch.cat([x, att2], 1)
        x = self.up2(x)
        x = torch.cat([x, att2], 1)
        x = self.up3(x)
        x = torch.cat([x, att2], 1)
        x = self.up4(x)

        return x, a
    
    def attention(self, bat_features):
        # d_k = self.feat_dim  # dim of single feature
        #query = features.view([self.k_inst, d_k/self.split_num])  
        scores = F.softmax(self.projection(bat_features), dim=-1) # [batchsize/k_inst, k_inst, 1]
        return torch.matmul(scores.permute(0, 1, 3, 2), bat_features)
    
class _netCC_4(nn.Module):
    def __init__(self, att_dim, cla_num=50, in_dim=2048, out_dim=2048):
        super(_netCC_4, self).__init__()
        self.att_dim = att_dim
        self.cla_num = cla_num
        self.down1 = nn.Sequential(nn.Linear(in_dim + self.att_dim, 1024), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(1024 + self.att_dim, 512), nn.ReLU())
        self.down3 = nn.Sequential(nn.Linear(512 + self.att_dim, 256), nn.ReLU())
        self.down4 = nn.Sequential(
            nn.Linear(256 + self.att_dim, self.att_dim), nn.ReLU()
        )

        self.up1 = nn.Sequential(nn.Linear(self.att_dim, 256), nn.ReLU())
        self.up2 = nn.Sequential(nn.Linear(256 + self.att_dim, 512), nn.ReLU())
        self.up3 = nn.Sequential(nn.Linear(512 + self.att_dim, 1024), nn.ReLU())
        self.up4 = nn.Sequential(nn.Linear(1024 + self.att_dim, out_dim), nn.ReLU())
        self.attention = nn.Sequential(
            nn.Linear(out_dim + self.cla_num, self.cla_num), nn.ReLU()
        )
        self.softmax = nn.LogSoftmax()

    def forward(self, feature1, att2, distance, feature2):
        x = torch.cat([feature1, att2], 1)
        x = self.down1(x)
        x = torch.cat([x, att2], 1)
        x = self.down2(x)
        x = torch.cat([x, att2], 1)
        x = self.down3(x)
        x = torch.cat([x, att2], 1)
        a = self.down4(x)
        x = self.up1(a)
        x = torch.cat([x, att2], 1)
        x = self.up2(x)
        x = torch.cat([x, att2], 1)
        x = self.up3(x)
        x = torch.cat([x, att2], 1)
        x = self.up4(x)
        # print(x.shape)
        # print(feature2.shape)
        xx = torch.cat([x, feature2], 0)
        dis = torch.cat([distance, distance], 0)
        new = torch.cat([xx, dis], 1)
        return x, a, self.softmax(self.attention(new))


class _netCC_5(nn.Module):
    # _netCC_3 的折叠对称版本
    def __init__(self, att_dim, in_dim=2048, out_dim=2048):
        super(_netCC_5, self).__init__()
        self.att_dim = att_dim
        self.down1 = nn.Sequential(nn.Linear(in_dim + self.att_dim, 1024), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(1024 + self.att_dim, 512), nn.ReLU())
        self.down3 = nn.Sequential(nn.Linear(512 + self.att_dim, 256), nn.ReLU())
        self.down4 = nn.Sequential(
            nn.Linear(256 + self.att_dim, self.att_dim), nn.ReLU()
        )

        self.up1 = nn.Sequential(nn.Linear(self.att_dim, 256), nn.ReLU())
        self.up2 = nn.Sequential(nn.Linear(256 + self.att_dim, 512), nn.ReLU())
        self.up3 = nn.Sequential(nn.Linear(512 + self.att_dim, 1024), nn.ReLU())
        self.up4 = nn.Sequential(nn.Linear(1024 + self.att_dim, out_dim), nn.ReLU())

        self.down1_2 = nn.Sequential(nn.Linear(in_dim + self.att_dim, 1024), nn.ReLU())
        self.down2_2 = nn.Sequential(nn.Linear(1024 + self.att_dim, 512), nn.ReLU())
        self.down3_2 = nn.Sequential(nn.Linear(512 + self.att_dim, 256), nn.ReLU())
        self.down4_2 = nn.Sequential(
            nn.Linear(256 + self.att_dim, self.att_dim), nn.ReLU()
        )

        self.up1_2 = nn.Sequential(nn.Linear(self.att_dim, 256), nn.ReLU())
        self.up2_2 = nn.Sequential(nn.Linear(256 + self.att_dim, 512), nn.ReLU())
        self.up3_2 = nn.Sequential(nn.Linear(512 + self.att_dim, 1024), nn.ReLU())
        self.up4_2 = nn.Sequential(nn.Linear(1024 + self.att_dim, out_dim), nn.ReLU())

    def forward(self, feature1, att1, att2):
        x = torch.cat([feature1, att2], 1)
        x = self.down1(x)
        x = torch.cat([x, att2], 1)
        x = self.down2(x)
        x = torch.cat([x, att2], 1)
        x = self.down3(x)
        x = torch.cat([x, att2], 1)
        a2 = self.down4(x)
        x = self.up1(a2)
        x = torch.cat([x, att2], 1)
        x = self.up2(x)
        x = torch.cat([x, att2], 1)
        x = self.up3(x)
        x = torch.cat([x, att2], 1)
        x = self.up4(x)

        y = torch.cat([x, att1], 1)
        y = self.down1(y)
        y = torch.cat([y, att1], 1)
        y = self.down2(y)
        y = torch.cat([y, att1], 1)
        y = self.down3(y)
        y = torch.cat([y, att1], 1)
        a1 = self.down4(y)
        y = self.up1(a1)
        y = torch.cat([y, att1], 1)
        y = self.up2(y)
        y = torch.cat([y, att1], 1)
        y = self.up3(y)
        y = torch.cat([y, att1], 1)
        y = self.up4(y)

        # x: target sample
        # a2: target att
        # y: source sample
        # a1: source att
        return x, a2, y, a1


# def attention(query, key, value, mask=None, dropout=None):

#     "Implementation of Scaled dot product attention"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     # print('q',query.size())
#     # print('s',scores.size())
#     # print('m',mask.size())
#     if mask is not None:
#         print(mask.size())
#         print(scores.size())
#         scores = scores.masked_fill(mask == 0, -1e9)
#         # scores = scores.masked_fill(mask == 0, 0)
#     p_attn = F.softmax(scores, dim=-1)

#     # p_attn = p_attn.masked_fill(p_attn == 0.0100,0) ###
#     # # print(p_attn.transpose(1, 2)[0][99])

#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn

class conv_model(nn.Module):
    def __init__(self, attr_dim, output_dim):
        super(conv_model, self).__init__()
        self.attr_dim = attr_dim
        self.output_dim = output_dim
        
        channel_num = 5
        self.conv1 = nn.Conv2d(1, 50, kernel_size=1)
        self.conv2 = nn.Conv2d(1, channel_num, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channel_num)
        self.conv_out_dim = 50 * (attr_dim)* channel_num
        self.fc = nn.Linear(self.conv_out_dim, self.output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 
        x = x.view(-1, 1, 50, self.attr_dim)  # batch, channel, height , width
        x = F.relu(self.bn1(self.conv2(x)))
        #x = x.view(-1, 50, 50, self.attr_dim)  
        #x = torch.mean(x, dim=1) # average the feature map

        x = x.view(-1, self.conv_out_dim)  # flatten

        x = self.fc(x)

        return x    

# Encoder
class Encoder(nn.Module):
    def __init__(self, _in_dim = 2048, _out_dim = 85):

        super(Encoder, self).__init__()
        latent_size = _out_dim 
        self.fc1 = nn.Linear(_in_dim+latent_size, 1024)  # cat(visual, attribute)
        self.fc2 = nn.Linear(1024, latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size * 2, latent_size)
        self.linear_log_var = nn.Linear(latent_size * 2, latent_size)
        
    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


# Decoder/Generator
class Generator(nn.Module):
    def __init__(self, _in_dim = 85, _out_dim = 2048):

        super(Generator, self).__init__()
        self.fc1 = nn.Linear(_in_dim * 2, 1024)  # cat(visual, attribute)
        self.fc2 = nn.Linear(1024, _out_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

    def _forward(self, z, c=None): 
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc2(x1))
        self.out = x1
        return x

    def forward(self, z, a1=None, c=None, feedback_layers=None): # c: attribute
        if feedback_layers is None:
            return self._forward(z, c)
        # else:
        #     z = torch.cat((z, c), dim=-1)
        #     x1 = self.lrelu(self.fc1(z))
        #     feedback_out = x1 + a1 * feedback_layers
        #     x = self.sigmoid(self.fc3(feedback_out))
        #     return x

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()