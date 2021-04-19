import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.nz = nz
        net = []
        channels_in = [self.nz, ngf * 8, ngf * 4, ngf * 2, ngf]
        channels_out = [ngf * 8, ngf * 4, ngf * 2, ngf, nc]
        active = ["R", "R", "R", "R", "tanh"]
        stride = [1, 2, 2, 2, 2]
        padding = [0, 1, 1, 1, 1]
        for i in range(len(channels_in)):
            net.append(nn.ConvTranspose2d(in_channels=channels_in[i], out_channels=channels_out[i], kernel_size=4, stride=stride[i], padding=padding[i], bias=False))
            if active[i] == "R":
                net.append(nn.BatchNorm2d(num_features=channels_out[i]))
                net.append(nn.ReLU())
            elif active[i] == "tanh":
                net.append(nn.Tanh())

        self.generator = nn.Sequential(*net)
        self.weight_init()

    def forward(self, input):
        output = self.generator(input)
        return output

    def weight_init(self):
        for m in self.generator.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    def __init__(self, nz=100, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        net = []
        # 1:预先定义
        channels_in = [nc, ndf, ndf * 2, ndf * 4, ndf * 8]
        channels_out = [ndf, ndf * 2, ndf * 4, ndf * 8, 1]
        padding = [1, 1, 1, 1, 0]
        active = ["LR", "LR", "LR", "LR", "sigmoid"]
        for i in range(len(channels_in)):
            net.append(nn.Conv2d(in_channels=channels_in[i], out_channels=channels_out[i], kernel_size=4, stride=2, padding=padding[i], bias=False))
            if i == 0:
                net.append(nn.LeakyReLU(0.2))
            elif active[i] == "LR":
                net.append(nn.BatchNorm2d(num_features=channels_out[i]))
                net.append(nn.LeakyReLU(0.2))
            elif active[i] == "sigmoid":
                net.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*net)
        self.weight_init()

    def forward(self, x):
        out = self.discriminator(x)
        out = out.view(x.size(0), -1)
        return out

    def weight_init(self):
        for m in self.generator.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)