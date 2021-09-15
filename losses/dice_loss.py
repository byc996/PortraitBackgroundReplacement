# _*_ coding: utf-8 _*_
"""
# @Time : 8/27/2021 5:12 PM
# @Author : byc
# @File : dice_loss.py
# @Description :
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    # soft dice loss
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pred = torch.sigmoid(predict).view(num, -1)
        targ = target.view(num, -1)

        intersection = (pred * targ).sum()
        union = (pred + targ).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union +self.epsilon)

        return score


if __name__ == "__main__":

    fake_out = torch.tensor([7, 7, -5, -5], dtype=torch.float32)
    fake_label = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    loss_f = DiceLoss()
    loss = loss_f(fake_out, fake_label)

    print(loss)