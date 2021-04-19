import torch
from torch.optim import Adam
from torch.backends import cudnn
from dcgan import Generator, Discriminator
import numpy as np
import os
import cv2
import time


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        # 训练集DataLoader
        self.train_loader = train_loader
        # 测试集DataLoader
        self.test_loader = test_loader
        # config配置
        self.config = config
        # 展示信息epoch次数
        self.show_every = config.show_every
        # 学习率衰退epoch数
        self.lr_decay_epoch = [
            15,
        ]
        # 创建模型
        self.build_model()
        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()
        # 进入test模式
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            # 载入预训练模型并放入相应位置
            if self.config.cuda:
                self.netG.load_state_dict(torch.load(self.config.model))
                self.netD.load_state_dict(torch.load(self.config.model))
            else:
                self.netG.load_state_dict(torch.load(self.config.model, map_location='cpu'))
                self.netD.load_state_dict(torch.load(self.config.model, map_location='cpu'))

    # 打印网络信息和参数数量
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # 建立模型
    def build_model(self):
        self.netG = Generator(nz=self.config.nz, ngf=self.config.ngf, nc=self.config.nc)
        self.netD = Discriminator(nz=self.config.nz, ndf=self.config.ndf, nc=self.config.nc)
        # 是否将网络搬运至cuda
        if self.config.cuda:
            self.netG = self.net.cuda()
            self.netD = self.net.cuda()
            cudnn.benchmark = True
        # self.net.train()
        # 设置eval状态
        self.netG.eval()  # use_global_stats = True
        self.netD.eval()
        # 载入预训练模型或自行训练模型
        if self.config.load == '':
            self.netG.load_state_dict(torch.load(self.config.pretrained_model))
            self.netD.load_state_dict(torch.load(self.config.pretrained_model))
        else:
            self.netG.load_state_dict(torch.load(self.config.load))
            self.netD.load_state_dict(torch.load(self.config.load))

        # 设置优化器
        self.optimizerD = Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2), weight_decay=self.config.wd)
        self.optimizerG = Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2), weight_decay=self.config.wd)
        # 打印网络结构
        self.print_network(self.netG, 'Generator Structure')
        self.print_network(self.netD, 'Discriminator Structure')

    # testing状态
    def test(self):
        # 训练模式
        mode_name = 'enhanced'
        # 开始时间
        time_s = time.time()
        # images数量
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            # 获取image数据和name
            phone_image, _, name = data_batch['phone_image'], data_batch['dslr_image'], data_batch['name']
            # testing状态
            with torch.no_grad():
                # 获取tensor数据并搬运指定设备
                images = torch.Tensor(phone_image)
                if self.config.cuda:
                    images = images.cuda()
                # 预测值
                preds = self.netG(images).cpu().data.numpy()
                # 创建image
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png'), preds)
        # 结束时间
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    # training状态
    def train(self):
        for epoch in range(self.config.epochs):
            for i, data_batch in enumerate(self.train_loader):
                # 获取image数据和name
                phone_image, _, _ = data_batch['phone_image'], data_batch['dslr_image'], data_batch['name']
                # Adversarial ground truths
                valid = torch.Tensor(phone_image.size(0), 1).fill_(1.0)
                fake = torch.Tensor(phone_image.size(0), 1).fill_(0.0)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizerG.zero_grad()

                # Sample noise as generator input
                z = torch.Tensor(np.random.normal(0, 1, (phone_image.shape[0], self.config.nz)))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizerG.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizerD.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(phone_image), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizerD.step()

                # 展示此时信息
                if i % (self.show_every // self.config.batch_size) == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, self.config.epochs, i, len(self.train_loader), d_loss.item(), g_loss.item()))
                    print('Learning rate: ' + str(self.config.lr))

            # 保存训练模型
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.netG.state_dict(), '%s/models/generator/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
                torch.save(self.netD.state_dict(), '%s/models/discriminator/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            # 学习率衰退
            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                # 设置优化器
                self.optimizerG = Adam(filter(lambda p: p.requires_grad, self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2), weight_decay=self.config.wd))
                self.optimizerD = Adam(filter(lambda p: p.requires_grad, self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2), weight_decay=self.config.wd))

        # 保存训练模型
        torch.save(self.net.state_dict(), '%s/models/generator/final.pth' % self.config.save_folder)
        torch.save(self.net.state_dict(), '%s/models/discriminator/final.pth' % self.config.save_folder)
