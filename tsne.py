# 用于画tsne figures 
# 本代码在本机运行，不是在202.112.113.57上运行的
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.io as sio

dataset = 'APY'
real_file = sio.loadmat('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/real_test_data.mat'.format(dataset))
real_data = real_file['data']
real_label = real_file['label']
R = real_file['tsnedata']
# R = TSNE(n_components=2).fit_transform(real_data)
# sio.savemat('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/real_test_data.mat'.format(dataset),{'data': real_data, 'label': real_label, 'tsnedata': R})


fake_file = sio.loadmat('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/fake_test_data.mat'.format(dataset))
fake_data = fake_file['data']
fake_label = fake_file['label']
F = fake_file['tsnedata']
# F = TSNE(n_components=2).fit_transform(fake_data)
# sio.savemat('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/fake_test_data.mat'.format(dataset),{'data': fake_data, 'label': fake_label, 'tsnedata': F})



all_data = np.vstack((real_data, fake_data))
all_label = np.hstack((real_label, fake_label))
ALL = TSNE(n_components=2).fit_transform(all_data)
sio.savemat('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/all_test_data.mat'.format(dataset),{'data': all_data, 'label': all_label, 'tsnedata': ALL})



# apy_classes = ['cow','horse','motorbike','person','pottedplant','sheep','train','tvmonitor','donkey','goat','jetski','statue']


# real tsne
fig, ax = plt.subplots()
for c in np.unique(real_label):

    ds = R[np.where(real_label[0]==c)[0]]
    x = ds[:,0]
    y = ds[:,1]
    # if c!='person':
    ax.scatter(x, y, s=10, alpha=0.5, edgecolors='none')
    # print(ds)

# ax.legend(ncol=2, loc='lower right')
plt.title('{}-REAL'.format('aPY'))
plt.savefig('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/real.jpg'.format(dataset))
plt.savefig('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/real.png'.format(dataset))
plt.show()


# fake tsne
fig2, ax2 = plt.subplots()
for c in np.unique(fake_label):

    ds = F[np.where(fake_label[0]==c)[0]]
    x = ds[:,0]
    y = ds[:,1]
    # if c!='person':
    ax2.scatter(x, y, s=10, alpha=0.5, edgecolors='none')
# ax2.legend(ncol=2, loc='lower right')
plt.title('{}-GENERATED'.format('aPY'))
plt.savefig('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/fake.jpg'.format(dataset))
plt.savefig('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/fake.png'.format(dataset))
plt.show()


# all tsne
fig3, ax3 = plt.subplots()
for c in np.unique(all_label):

    ds = ALL[np.where(all_label[0]==c)[0]]
    x = ds[:,0]
    y = ds[:,1]
    # if c!='person':
    ax3.scatter(x, y, s=10, alpha=0.5, edgecolors='none')
# ax3.legend(ncol=2, loc='lower right')
plt.title('{}-ALL'.format(dataset))
plt.savefig('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/all.jpg'.format(dataset))
plt.savefig('/Users/liujinlu/myfile/MyPaper/2019InformationSciences/imgs/tsne/{}/all.png'.format(dataset))
plt.show()


