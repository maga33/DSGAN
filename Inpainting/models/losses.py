import torch
import torch.nn as nn
from torch.autograd import Variable
from .layer_util import *
import numpy as np
import functools

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids, normalize=False):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.normalize=normalize

        if self.normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            mean_img = torch.FloatTensor(1, 3, 1, 1)
            std_img = torch.FloatTensor(1, 3, 1, 1)
            for j in range(3):
                mean_img[:, j, :, :].fill_(mean[j])
                std_img[:, j, :, :].fill_(1.0/std[j])
            self.mean_img = Variable(mean_img.cuda())
            self.std_img = Variable(std_img.cuda())

    def normalize_input(self, input):
        input_normalized = (input - self.mean_img) * self.std_img
        return input_normalized

    def forward(self, x, y):
        if self.normalize:
            x,y = self.normalize_input(x), self.normalize_input(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

def compute_gan_loss(real_digits, fake_digits, loss_type=None,
        loss_at='None'):
    errD = None
    errG = None
    if loss_type == 'HINGE':
        if loss_at == 'D':
            errD =  torch.clamp(1 - real_digits, min=0) + \
                    torch.clamp(1 + fake_digits, min=0)

            errD = errD.mean()
        elif loss_at == 'G':
            errG = - fake_digits.mean()
    elif loss_type == 'KL':
        if loss_at == 'D':
            errD = F.logsigmoid(real_digits) +\
                    F.logsigmoid(- fake_digits)
            errD = - errD.mean()
        else:
            assert real_digits is None
            errG = - F.logsigmoid(fake_digits).mean()

    return errG, errD
