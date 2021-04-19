import argparse
import os

from load_dataset import get_loader
from solver import Solver


def main(config):
    # 选择使用模式
    if config.mode == 'train':
        # 获取DataLoader
        train_loader = get_loader(config)
        # 计数, 找到可以创建文件夹的名字
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        # 创建保存模型文件夹
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        # 在config中保存文件夹名称
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        # 开始进入training状态
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        # 获取DataLoader
        test_loader = get_loader(config, mode='test')
        # 如果文件路径不存在
        if not os.path.exists(config.test_fold):
            os.mkdir(config.test_fold)
        # 开始进入testing状态
        test = Solver(None, test_loader, config)
        test.test()
    else:
        # 非法输入
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    # 权重路径
    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'

    # 命令行参数初始化
    parser = argparse.ArgumentParser()

    # 图片类型
    parser.add_argument('--phone', type=str, required=True)
    # batch大小
    parser.add_argument('--batch_size', type=int, default=10)
    # 训练次数
    parser.add_argument('--epochs', type=int, default=30)
    # 训练次数
    parser.add_argument('--epoch_save', type=int, default=9)
    # 学习率
    parser.add_argument('--lr', type=float, default=5e-4)
    # 权重衰退
    parser.add_argument('--wd', type=float, default=0.0005)
    # Adam的beta1
    parser.add_argument('--beta1', type=float, default=0.5)
    # Adam的beta2
    parser.add_argument('--beta2', type=float, default=0.99)
    # 展示信息epoch次数
    parser.add_argument('--show_every', type=int, default=2000)
    # 保存训练模型文件夹
    parser.add_argument('--save_folder', type=str, default='./results')
    # orig模式
    parser.add_argument('--resolution', type=str, default='orig')
    # 是否使用cuda
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    # 载入自行训练模型
    parser.add_argument('--load', type=str, default='')
    # 载入预训练模型
    parser.add_argument('--pretrained_model', type=str, default='')
    # 载入test模型
    parser.add_argument('--model', type=str, default=None)
    # 测试结果保存文件夹
    parser.add_argument('--test_fold', type=str, default=None)
    # 生成器的输入向量长度
    parser.add_argument('--nz', type=int, default=100)
    # 生成器的feature map大小
    parser.add_argument('--ngf', type=int, default=64)
    # 鉴别器的feature map大小
    parser.add_argument('--ndf', type=int, default=64)
    # 训练图片的色彩空间通道
    parser.add_argument('--nc', type=int, default=3)
    # 权重
    parser.add_argument('--w_content', type=float, default=10)
    parser.add_argument('--w_color', type=float, default=0.5)
    parser.add_argument('--w_texture', type=float, default=1)
    parser.add_argument('--w_tv', type=float, default=200)
    # 数据集目录
    parser.add_argument('--dped_dir', type=str, default='dped/')
    # vgg网络权重
    parser.add_argument('--vgg_dir', type=str, default='vgg_pretrained/imagenet-vgg-verydeep-19.mat')
    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # 将参数保存至config
    config = parser.parse_args()
    print(config)

    # 训练模型文件夹路径是否存在
    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # 运行main函数
    main(config)
