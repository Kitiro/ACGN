# load trained model for obtaining projected semantic feature (e.g., 85 -> 2048)
import torch
from models import conv_model
from scipy import io as sio
model_path = '/home/zzc/exp/Hierarchically_Learning_The_Discriminative_Features_For_Zero_Shot_Learning/output/AWA2/Model_Beta-1e-05_GenAttr-0.pth' # bestAcc on ZSL: 75

mat_path = 'data/AwA2/att_splits.mat'
mat = sio.loadmat(mat_path)
semantic = mat['att'].T

model = conv_model(attr_dim=semantic.shape[1], output_dim=2048)
model.load_state_dict(torch.load(model_path))
model.cuda()  

semantic = torch.tensor(semantic).float().reshape(semantic.shape[0], 1, 1, semantic.shape[1]).cuda()
                  
mapped_semantic = model(semantic)
mapped_semantic = mapped_semantic.cpu().detach().numpy().squeeze()
sio.savemat('data/AwA2/mapped_semantic.mat', {'mapped_feature':mapped_semantic})