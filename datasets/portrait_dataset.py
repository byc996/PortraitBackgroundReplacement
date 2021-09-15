# _*_ coding: utf-8 _*_
"""
# @Time : 8/27/2021 3:26 PM
# @Author : byc
# @File : portrait_dataset.py
# @Description :
"""
import os
import random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class PortraitDataset2000(Dataset):

    cls_num = 2
    names = ['bg', 'portrait']

    def __init__(self, root_dir, transform=None, in_size=224):
        super(PortraitDataset2000, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.label_path_list = list()
        self.in_size = in_size

        self._get_img_path()

    def __getitem__(self, item):
        # step1: read image file to ndarray
        path_label = self.label_path_list[item]
        path_img = path_label[:-10] + '.png'
        img_bgr = cv2.imread(path_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        msk_rgb = cv2.imread(path_label)

        # step2: image preprocess
        if self.transform:
            transformed = self.transform(image=img_rgb, mask=msk_rgb)
            img_rgb = transformed['image']
            msk_rgb = transformed['mask']

        # step3: image, mask -> tensor
        img_rgb = img_rgb.transpose((2, 0, 1))  # hwc --> chw
        img_chw_tensor = torch.from_numpy(img_rgb).float()

        msk_gray = msk_rgb[:, :, 0]  # hwc -> hw
        msk_gray = msk_gray / 255.   # [0, 255] scale -> [0, 1]
        label_tensor = torch.tensor(msk_gray, dtype=torch.float)

        return img_chw_tensor, label_tensor

    def __len__(self):
        return len(self.label_path_list)

    def _get_img_path(self):
        file_list = os.listdir(self.root_dir)
        file_list = list(filter(lambda x: x.endswith('_matte.png'), file_list))
        path_list = [os.path.join(self.root_dir, name) for name in file_list]
        random.shuffle(path_list)
        if len(path_list) == 0:
            raise Exception('\nroot_dir: {} is a empty dir!'.format(self.root_dir))
        self.label_path_list = path_list


if __name__ == '__main__':

    por_dir = r'G:\DeepShare\Data\Portrait-dataset-2000\training'

    por_set = PortraitDataset2000(por_dir)
    train_loader = DataLoader(por_set, batch_size=1, shuffle=True, num_workers=0)
    print(len(por_set))
    for i, sample in enumerate(train_loader):
        img, label = sample
        print(img.shape)