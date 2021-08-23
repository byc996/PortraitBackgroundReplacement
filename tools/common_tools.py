# _*_ coding: utf-8 _*_
"""
# @Time : 8/22/2021 12:15 PM
# @Author : byc
# @File : common_tools.py
# @Description : common tool functions
"""
import random
import numpy as np
import torch


def setup_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True