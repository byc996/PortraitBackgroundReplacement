# _*_ coding: utf-8 _*_
"""
# @Time : 8/22/2021 12:13 PM
# @Author : byc
# @File : protrait_config.py
# @Description : segmentation configs for portrait dataset
"""

import torch
from easydict import EasyDict
import albumentations as A

cfg = EasyDict()

# cfg.loss_type = "BCE"
# cfg.loss_type = "BCE&dice"
cfg.loss_type = "dice"
# cfg.loss_type = "focal"
cfg.focal_alpha = 0.5
cfg.focal_gamma = 0.5  # 0.5， 2， 5， 10

# warmup cosine decay
cfg.is_warmup = True
cfg.warmup_epochs = 1
cfg.lr_final = 1e-5
cfg.lr_warmup_init = 0.  # 是0. 没错

cfg.hist_grad = False

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.max_epoch = 40  # 50

cfg.is_fusion_data = True
cfg.is_ext_data = True
cfg.ext_num = 8500


# batch size
cfg.train_bs = 8   # 32
cfg.valid_bs = 4   # 24
cfg.workers = 16  # 16

# learning rate
cfg.lr_init = 0.01
cfg.factor = 0.1
cfg.milestones = [25, 45]
cfg.weight_decay = 5e-4
cfg.momentum = 0.9

cfg.log_interval = 10

cfg.bce_pos_weight = torch.tensor(1.)  # [2.91368023e+08 5.24631977e+08]  0.555

cfg.in_size = 512   # input size

norm_mean = (0.5, 0.5, 0.5)  # better than imagenet
norm_std = (0.5, 0.5, 0.5)

cfg.tf_train = A.Compose([
    A.ColorJitter(p=0.5, brightness=(0.4, 1.7), contrast=(0.6, 1.5), saturation=[0.9, 1.1], hue=[-0.1, 0.1]),
    A.HorizontalFlip(p=0.5),
    A.Resize(width=cfg.in_size, height=cfg.in_size),
    A.Normalize(norm_mean, norm_std),
])

cfg.tf_valid = A.Compose([
    A.Resize(width=cfg.in_size, height=cfg.in_size),
    A.Normalize(norm_mean, norm_std),
])


