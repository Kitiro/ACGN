
# Attentive Cross-class Generative Networks

环境：202.112.113.57, python3
数据：/data/liujinlu/xian_resnet101/xlsa17/data
基础文件：dataset.py, models.py
主流程文件，可直接运行：train_gan.py, train_nongan.py
画图文件，用于各个实验作图，基本上都是在本机上做的：entropy.py, figure.py, tsne.py, save_generated_data.py, top5_samples.py, data_for_tsne.py
其他文件：基本上仅作为尝试，没有采用，效果不好
实验结果.xlsx： gan（3，1）是指用train_gan.py，采用3个相似类，每个相似类中取1个相似样例的结果；(random)为取1个相似类，每个相似类中随机取1个相似样例的结果

python train_gan_attention_for_class.py --gpu 0 --batchsize 256 --x_w 0
python train_gan_attention_for_class.py --gpu 0 --batchsize 256 --x_w 1 --k_class 2 --k_inst 2