###
 # @Author: Kitiro
 # @Date: 2021-01-30 12:51:03
 # @LastEditTime: 2021-08-02 23:34:30
 # @LastEditors: Kitiro
 # @Description: 
 # @FilePath: /liujinlu_code/gan_attention_test.sh
### 
att='10 100'
x='100 500'
cent='1 10 50'
dataset='AwA2 apy cub sun'
inst='1 2 4'
cla='2 4'
ratio='0.2 0.5 0.8'
split_num='4 8 10'

for s in $split_num
do
    for c in $cla
    do
        for i in $inst
        do  
            
            # python train_gan_attention_for_class.py --gpu 0 --k_cla 1 --k_inst 4 --endEpoch 2000 --att_w $i --x_w $j --cent_w $k --batchsize 200 --dataset AwA2 --gzsl &
            # python train_gan_modifyD.py --gpu 0 --k_cla 1 --k_inst 1 --endEpoch 2000 --att_w $i --x_w $j --cent_w $k --batchsize 200 --dataset AwA2 --gzsl &
            python test_attention.py --gpu 1 --k_cla $c --k_inst $i --selected_ratio 0.8 --endEpoch 3000 --att_w 10 --x_w 10 --cent_w 10 --batchsize 256 --dataset AwA2 --gzsl --split_num $s &
        done
        wait
    done
done
