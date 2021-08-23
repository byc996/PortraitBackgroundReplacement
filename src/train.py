# _*_ coding: utf-8 _*_
"""
# @Time : 8/22/2021 12:02 PM
# @Author : byc
# @File : train.py
# @Description : main code for model training
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))  # add project dir to env path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config.protrait_config import cfg
from tools.common_tools import setup_seed


setup_seed(20210801)

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, help='learning rate', type=float)
parser.add_argument('--max_epoch', default=None, type=int)
parser.add_argument('--train_bs', default=0, type=int)
parser.add_argument('--data_root_dir', default='', help='path to your dataset')
parser.add_argument('--ext_dir', default='', help='path to extra dataset')
parser.add_argument('--fusion_dir', default='', help='path to fusion dataset')

args = parser.parse_args()
cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.train_bs if args.train_bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch


if __name__ == '__main__':
    pass