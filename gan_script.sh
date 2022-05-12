###
 # @Author: Kitiro
 # @Date: 2021-01-30 12:51:03
 # @LastEditTime: 2021-08-03 23:26:10
 # @LastEditors: Kitiro
 # @Description: 
 # @FilePath: /liujinlu_code/gan_script.sh
### 
att='100'
x_w='10 100 500'
cent='1 10 50'
dataset='AwA2 apy cub sun'
inst='2 4'
cla='2 4'
ratio='0.5 0.8'
split_num='4 8 16'
scale='0.1 0.001 0.0001'

# for s in $split_num
# do
#     for r in $ratio
#     do
#         for i in $inst
#         do  
            
#             # python train_gan_attention_for_class.py --gpu 0 --k_cla 1 --k_inst 4 --endEpoch 2000 --att_w $i --x_w $j --cent_w $k --batchsize 200 --dataset AwA2 --gzsl &
#             # python train_gan_modifyD.py --gpu 0 --k_cla 1 --k_inst 1 --endEpoch 2000 --att_w $i --x_w $j --cent_w $k --batchsize 200 --dataset AwA2 --gzsl &
#             python train_gan_attention_for_class_more_unseen.py --gpu 0 --k_cla 1 --k_inst $i --selected_ratio $r --endEpoch 3000 --att_w 100 --x_w 50 --cent_w 10 --batchsize 256 --dataset AwA2 --gzsl --split_num $s &
#         done
#         wait
#     done
# done

for d in $dataset
do
    for x in $x_w
    do
        for c in $cent 
        do
            for i in $inst
            do  
                python train_gan_attention_for_class_with_noise.py --gpu 0 --scale 0.001 --k_cla 1 --k_inst $i --selected_ratio 0.8 --endEpoch 3000 --att_w 10 --x_w $x --cent_w $c --batchsize 512 --dataset $d --gzsl --split_num 8 &
            done
            wait
        done
    done
done
# nohup python train_gan_attention_for_class.py --gpu 0 --k_cla 1 --k_inst 4 --endEpoch 1000 --att_w 100 --x_w 10 --cent_w 50 --batchsize 512 --dataset AwA2 --gzsl > attention_result.txt 2>&1 &


# 69.45
# python train_nongan.py --gpu 0 --k_cla 1 --k_inst 4 --att_w 10 --x_w 15 --cent_w 3


# 66.47
# python train_nongan_attention.py --gpu 0 --k_cla 1 --k_inst 4 --att_w 500 --x_w 20 --cent_w 4

# 70.33
# python train_nongan_attention.py --gpu 0 --k_cla 1 --k_inst 4 --att_w 500 --x_w 20 --cent_w 10
