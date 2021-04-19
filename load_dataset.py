import torch
from torch.utils.data import Dataset
import os
import imageio
import numpy as np
from torch.utils import data
from utils import get_resolutions, get_specified_res


class LoadData(Dataset):
    def __init__(self, phone, dped_dir, dataset_size, image_size, test=False):

        if test:
            self.directory_phone = os.path.join(dped_dir, 'test', phone)
            self.directory_dslr = os.path.join(dped_dir, 'test', 'canon')
        else:
            self.directory_phone = os.path.join(dped_dir, 'train', phone)
            self.directory_dslr = os.path.join(dped_dir, 'train', 'canon')

        self.dataset_size = dataset_size
        self.test = test
        self.image_size = image_size

    def __len__(self):
        # 数据集长度
        return self.dataset_size

    def __getitem__(self, idx):
        # 初始图片
        phone_image = np.asarray(imageio.imread(os.path.join(self.directory_phone, str(idx) + '.png')))
        phone_image = np.float16(np.reshape(phone_image, [1, self.image_size])) / 255
        phone_image = torch.from_numpy(phone_image.transpose((2, 0, 1)))
        # 高清图片
        dslr_image = np.asarray(imageio.imread(os.path.join(self.directory_dslr, str(idx) + '.png')))
        dslr_image = np.float16(np.reshape(dslr_image, [1, self.image_size])) / 255
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
        # 每提取100张, 进行输出提示
        if idx % 100 == 0:
            print(str(round(idx * 100 / self.dataset_size)) + "% done", end="\r")
        return {'phone_image': phone_image, 'dslr_image': dslr_image, 'name': self.directory_phone + str(idx)}


# 获取DataLoader
def get_loader(config, mode='train', pin=False):
    res_sizes = get_resolutions()
    _, _, IMAGE_SIZE = get_specified_res(res_sizes, config.phone, config.resolution)
    # 定义不同模式下的DataLoader
    if mode == 'train':
        shuffle = True
        dataset = LoadData(config.phone, config.dped_dir, IMAGE_SIZE)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    else:
        shuffle = False
        dataset = LoadData(config.phone, config.dped_dir, IMAGE_SIZE, test=True)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    return data_loader
