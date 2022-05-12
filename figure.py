# coding:utf8
# 本文件用于画图表，每段代码单独运行

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2, 1.01*height, '%d' % int(height))

### APY_RE ####
num_list = [2,3,9,6]
label_list = [1,2,3,4]
plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=.25)
a = plt.bar(range(len(num_list)), num_list, width = 0.5, color='darkblue', tick_label=label_list)
# autolabel(a)
plt.xlabel(r'$\lfloor RE\rfloor$')
plt.ylabel('Number of Classes')
# plt.title('aPY')
plt.legend()
plt.show()


#### AWA1_RE  ######
num_list = [5,5,18,12]
label_list = [1,2,3,4]
a = plt.bar(range(len(num_list)), num_list, width = 0.5, color='darkblue', tick_label=label_list)
# autolabel(a)
plt.xlabel(r'$\lfloor RE\rfloor$')
plt.ylabel('Number of Classes')
# plt.title('AWA1')
plt.legend()
plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=.25)
plt.show()


#### AWA2_RE  ######
num_list = [1,0,1,1,1,0,1,0,1,1,\
            3,2,2,1,2,1,2,1,4,5,\
            3,1,1,1,1,0,2,0,1]
label_list = [1,2,3,4,5,6,7,8,9,10,\
            11,12,13,14,15,16,17,18,19,20,\
            21,22,23,24,25,26,27,28,29]
a = plt.bar(range(len(num_list)), num_list, width = 0.4, color='darkblue', tick_label=label_list)
# autolabel(a)
plt.xlabel(r'$\lfloor RE\rfloor$')
plt.ylabel('Number of Classes')
# plt.title('AWA2')
plt.legend()
plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=.25)
plt.show()


#### CUB_RE  ######
num_list = [13,43,82,12]
label_list = [1,2,3,4]
a = plt.bar(range(len(num_list)), num_list, width = 0.5, color='darkblue', tick_label=label_list)
# autolabel(a)
plt.xlabel(r'$\lfloor RE\rfloor$')
plt.ylabel('Number of Classes')
# plt.title('CUB')
plt.legend()
plt.grid(axis='y', linestyle='--', which='major', color='grey', alpha=.25)
plt.show()



#### SUN_RE  ######
num_list = [32,250,363]
label_list = [1,2,3]
fig,axes=plt.subplots()
axes.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
axes.bar(range(len(num_list)), num_list, width = 0.4, color='darkblue', tick_label=label_list)
# autolabel(a)
plt.xlabel(r'$\lfloor RE\rfloor$')
plt.ylabel('Number of Classes')
# plt.title('SUN')
plt.legend()

plt.show()



#####  CUB Source Class  ########
cla_num = [1,2,3,4]
cub_gan = [51.65, 52.02, 52.27, 51.12]
cub_nongan = [52.67, 52.61, 52.46, 52.42]

cubplot1, = plt.plot(cla_num, cub_gan, color='orangered',linestyle='--')
cubplot2, = plt.plot(cla_num, cub_nongan, color='green')
plt.xlabel('Number of Source Class', fontsize=19)
plt.ylabel('Accuracy(%)', fontsize=20)
# plt.title('CUB', fontsize=20)
plt.ylim(40,65)
plt.grid(axis='y')
plt.xticks(cla_num)
plt.legend([cubplot1, cubplot2], ['ACGN', 'NACGN'], fontsize=15,loc='upper right')

for a, b in zip([1,2,3,4], cub_gan):
    plt.text(a, b, b, ha='center', va='top',fontsize=12)
for a, b in zip([1,2,3,4], cub_nongan):
    plt.text(a, b, b, ha='center', va='bottom',fontsize=12)

plt.show()


#####  CUB Source Sample  ########
cla_num = [1,2,3,4]
cub_gan = [51.65, 51.89, 52.44, 52.02]
cub_nongan = [52.67, 52.15, 52.16, 52.45]

cubplot1, = plt.plot(cla_num, cub_gan, color='orangered',linestyle='--')
cubplot2, = plt.plot(cla_num, cub_nongan, color='green')
plt.xlabel('Number of Source Sample', fontsize=19)
plt.ylabel('Accuracy(%)', fontsize=20)
# plt.title('CUB', fontsize=20)
plt.ylim(40,65)
plt.grid(axis='y')
plt.xticks(cla_num)
plt.legend([cubplot1, cubplot2], ['ACGN', 'NACGN'], fontsize=15,loc='upper right')

for a, b in zip([1,2,3,4], cub_gan):
    plt.text(a, b, b, ha='center', va='top',fontsize=12)
for a, b in zip([1,2,3,4], cub_nongan):
    plt.text(a, b, b, ha='center', va='bottom',fontsize=12)

plt.show()

#####  APY Source Class #########
cla_num = [1,2,3,4]
apy_gan = [41.1, 39.12, 36.72, 36.03]
apy_nongan = [39.98, 31.79, 34.93, 36.7]

apyplot1, = plt.plot(cla_num, apy_gan, color='orangered',linestyle='--')
apyplot2, = plt.plot(cla_num, apy_nongan, color='green')
plt.xlabel('Number of Source Class', fontsize=19)
plt.ylabel('Accuracy(%)', fontsize=20)
# plt.title('aPY', fontsize=20)
plt.ylim(20,50)
plt.grid(axis='y')
plt.xticks(cla_num)
plt.legend([apyplot1, apyplot2], ['ACGN', 'NACGN'], fontsize=15,loc='upper right')

for a, b in zip([1,2,3,4], apy_gan):
    plt.text(a, b, b, ha='center', va='bottom',fontsize=12)
for a, b in zip([1,2,3,4], apy_nongan):
    plt.text(a, b, b, ha='center', va='top',fontsize=12)

plt.show()


#####  APY Source Sample #########

cla_num = [1,2,3,4]
apy_gan = [41.1, 42.5, 44.43, 44.01]
apy_nongan = [39.98, 41.47, 40.83, 42.11]

apyplot1, = plt.plot(cla_num, apy_gan, color='orangered',linestyle='--')
apyplot2, = plt.plot(cla_num, apy_nongan, color='green')
plt.xlabel('Number of Source Sample', fontsize=19)
plt.ylabel('Accuracy(%)', fontsize=20)
# plt.title('aPY', fontsize=20)
plt.ylim(20,52)
plt.grid(axis='y')
plt.xticks(cla_num)
plt.legend([apyplot1, apyplot2], ['ACGN', 'NACGN'], fontsize=15,loc='upper right')

for a, b in zip([1,2,3,4], apy_gan):
    plt.text(a, b, b, ha='center', va='bottom',fontsize=12)
for a, b in zip([1,2,3,4], apy_nongan):
    plt.text(a, b, b, ha='center', va='top',fontsize=12)

plt.show()

##### SUN Source Class #############
cla_num = [1, 2, 3, 4]
sun_gan = [59.17, 57.5, 58.82, 58.68]
sun_nongan = [59.79, 59.17, 58.47, 59.38]

sunplot1, = plt.plot(cla_num, sun_gan, color='orangered',linestyle='--')
sunplot2, = plt.plot(cla_num, sun_nongan, color='green')
plt.xlabel('Number of Source Class', fontsize=19)
plt.ylabel('Accuracy(%)', fontsize=20)
# plt.title('SUN', fontsize=20)
plt.ylim(40,70)
plt.grid(axis='y')
plt.xticks(cla_num)
plt.legend([sunplot1, sunplot2], ['ACGN', 'NACGN'], fontsize=15,loc='upper right')

for a, b in zip([1,2,3,4], sun_gan):
    plt.text(a, b, b, ha='center', va='top',fontsize=12)
for a, b in zip([1,2,3,4], sun_nongan):
    plt.text(a, b, b, ha='center', va='bottom',fontsize=12)

plt.show()

##### SUN Source Sample #############
cla_num = [1, 2, 3, 4]
sun_gan = [59.17, 59.44, 58.82, 57.99]
sun_nongan = [59.79, 60, 59.93, 60.21]

sunplot1, = plt.plot(cla_num, sun_gan, color='orangered',linestyle='--')
sunplot2, = plt.plot(cla_num, sun_nongan, color='green')
plt.xlabel('Number of Source Sample', fontsize=19)
plt.ylabel('Accuracy(%)', fontsize=20)
# plt.title('SUN', fontsize=20)
plt.ylim(40,70)
plt.grid(axis='y')
plt.xticks(cla_num)
plt.legend([sunplot1, sunplot2], ['ACGN', 'NACGN'], fontsize=15,loc='upper right')

for a, b in zip([1,2,3,4], sun_gan):
    plt.text(a, b, b, ha='center', va='top',fontsize=12)
for a, b in zip([1,2,3,4], sun_nongan):
    plt.text(a, b, b, ha='center', va='bottom',fontsize=12)

plt.show()


