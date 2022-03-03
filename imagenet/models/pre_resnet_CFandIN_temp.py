import torch.nn as nn

import torch
import torch.nn.functional as functional
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable
import numpy as np

import torch.onnx
import netron
# from init import *
from random import random
import argparse



# __all__ = ['pre_resnet18', 'pre_resnet34', 'pre_resnet50', 'pre_resnet101',
#            'pre_resnet152']
__all__ = ['honet18_in', 'honet34_in', 'honet50_in', 'pre_act_resnet18_in', 'pre_act_resnet34_in', 'pre_act_resnet50_in']
# __all__ = ['HONet34_IN', 'HONet18_IN']
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
args = parser.parse_args()

global num_cla
num_cla = 1000

class BasicBlockWithDeathRate(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, death_rate=0., downsample=None):
        super(BasicBlockWithDeathRate, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.death_rate = death_rate

    def forward(self, x):

        if not self.training or torch.rand(1)[
            0] >= self.death_rate:  # 2nd condition: death_rate is below the upper bound
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            # ^ the same with Pre-ResNet
            if self.training:
                out /= (1. - self.death_rate)  # out = out/(1. - death_rate) ? maybe it is mutiplied by the rate before
        else:
            if self.stride == 1:
                out = Variable(torch.FloatTensor(x.size()).cuda().zero_(), requires_grad=False)
            else:

                size = list(x.size())
                size[-1] //= 2  # Maybe it is the Height (interger, devide)
                size[-2] //= 2  # Maybe it is the Width
                size[-3] *= 2  # Maybe Channel
                size = torch.Size(size)
                out = Variable(torch.FloatTensor(size).cuda().zero_(), requires_grad=False)  # all zero tensor
        return out


class BasicBlock_cifar(nn.Module):  # actually, this is the preact block
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock_cifar, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):  # Pre-ResNet
        out = self.bn1(x)  # wo BN
        # out = x # wo BN
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out


class HOBlock(nn.Module):  # actually, this is the preact block
    expansion = 1

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes, stride=1, k_ini=-9.0 / 5, fix_k=False,
                 stepsize=1, given_ks=[10, 10, 10, 10], downsample=None):
        super(HOBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.bn3 = nn.BatchNorm2d(planes)# 20210803
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.fix_k = fix_k
        if self.fix_k:
            self.k = k_ini
            self.a_0 = float(given_ks[0])
            self.a_1 = float(given_ks[1])
            self.a_2 = float(given_ks[2])
            self.b_0 = float(given_ks[3])
        else:
            self.k = nn.Parameter(torch.Tensor(1).uniform_(k_ini, k_ini))
        # self.ks = nn.ParameterList(torch.Tensor(1).uniform_(1.0, 1.1))
        # print('l_last_res_planes, last_res_planes, in_planes, planes', l_last_res_planes, last_res_planes, in_planes, planes)

        if not (self.last_res_planes == -1 or self.l_last_res_planes == -1):
            # if 1:
            if self.planes == 32:
                if in_planes == 16:
                    self.downsample_16_32_x = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_x')
                if self.last_res_planes == 16:
                    self.downsample_16_32_l = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_l')
                if self.l_last_res_planes == 16:
                    self.downsample_16_32_ll = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_ll')
            if self.planes == 64:
                if self.in_planes == 32:
                    self.downsample_32_64_x = Downsample_clean(32, 64, 2)
                if self.last_res_planes == 32:
                    self.downsample_32_64_l = Downsample_clean(32, 64, 2)
                if self.l_last_res_planes == 32:
                    self.downsample_32_64_ll = Downsample_clean(32, 64, 2)
            if self.planes == 128:
                if self.in_planes == 64:
                    self.downsample_64_128_x = Downsample_clean(64, 128, 2)
                if self.last_res_planes == 64:
                    self.downsample_64_128_l = Downsample_clean(64, 128, 2)
                if self.l_last_res_planes == 64:
                    self.downsample_64_128_ll = Downsample_clean(64, 128, 2)
            if self.planes == 256:
                if self.in_planes == 128:
                    self.downsample_128_256_x = Downsample_clean(128, 256, 2)
                if self.last_res_planes == 128:
                    self.downsample_128_256_l = Downsample_clean(128, 256, 2)
                if self.l_last_res_planes == 128:
                    self.downsample_128_256_ll = Downsample_clean(128, 256, 2)

    def forward(self, x, last_res, l_last_res):  # Pre-ResNet
        residual = x
        F_x_n = self.bn1(x)  # wo BN
        # F_x_n=x
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv1(F_x_n)
        F_x_n = self.bn2(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)
        # if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
        # print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()',  F_x_n.size()[1], residual.size()[1],last_res.size()[1],l_last_res.size()[1])
        # print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)
        if not (isinstance(last_res, int) or isinstance(l_last_res, int)):
            # print('HO')
            # if 1:
            if self.planes == 32:
                if self.in_planes == 16:
                    residual = self.downsample_16_32_x(residual)
                    # print('residual.size()', residual.size())
                if self.last_res_planes == 16:
                    last_res = self.downsample_16_32_l(last_res)
                # print('last_res.size()', last_res.size())
                if self.l_last_res_planes == 16:
                    l_last_res = self.downsample_16_32_ll(l_last_res)
                    # print('l_last_res.size()', l_last_res.size())
            if self.planes == 64:
                if self.in_planes == 32:
                    residual = self.downsample_32_64_x(residual)
                if self.last_res_planes == 32:
                    last_res = self.downsample_32_64_l(last_res)
                if self.l_last_res_planes == 32:
                    l_last_res = self.downsample_32_64_ll(l_last_res)
            if self.planes == 128:
                if self.in_planes == 64:
                    residual = self.downsample_64_128_x(residual)
                if self.last_res_planes == 64:
                    last_res = self.downsample_64_128_l(last_res)
                if self.l_last_res_planes == 64:
                    l_last_res = self.downsample_64_128_ll(l_last_res)
            if self.planes == 256:
                if self.in_planes == 128:
                    residual = self.downsample_128_256_x(residual)
                if self.last_res_planes == 128:
                    last_res = self.downsample_128_256_l(last_res)
                if self.l_last_res_planes == 128:
                    l_last_res = self.downsample_128_256_ll(l_last_res)
            if not self.fix_k:
                self.b_0 = (3 * self.k - 1) / (self.k * 2)
                self.a_0 = (3 * self.k + 3) / (self.k * 4)
                self.a_1 = -1 / (self.k)
                self.a_2 = (self.k + 1) / (4 * self.k)
                # print("trainable")
            x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(
                self.a_1, last_res) + torch.mul(self.a_2, l_last_res)
            # print('x', x[0][0][0][0])
            # print("self.a_0, self.a_1, self.a_2, self.b_0", self.a_0, self.a_1, self.a_2, self.b_0)
        else:
            # print('res')
            x = F_x_n
        # x = self.bn3(x)
        l_last_res = last_res
        last_res = residual  # x means the residual
        # residual = x
        return x, last_res, l_last_res, self.k


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self.stddev, requires_grad=False)
        return x


class Bottleneck_cifar(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_cifar, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out


class HoBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes, stride=1, k_ini=-9.0 / 5, fix_k=False,
                 stepsize=1, given_ks=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9]):
        super(HoBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.expansion = 4
        self.in_planes = in_planes
        self.planes = planes * self.expansion
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.fix_k = fix_k
        if self.fix_k:
            self.k = k_ini
            self.a_0 = float(given_ks[0])
            self.a_1 = float(given_ks[1])
            self.a_2 = float(given_ks[2])
            self.b_0 = float(given_ks[3])
        else:
            self.k = nn.Parameter(torch.Tensor(1).uniform_(k_ini, k_ini))
        # self.ks=nn.ParameterList(torch.Tensor(1).uniform_(1.0, 1.1))
        # self.downsample_16_64_res = Downsample_clean(16, 64, 1)
        # if not (last_res_planes == -1 and l_last_res_planes == -1):
        # if 1:
        if not (last_res_planes == -1 or l_last_res_planes == -1):
            if self.planes == 32:
                if in_planes == 16:
                    self.downsample_16_32_x = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_x')
                if last_res_planes == 16:
                    self.downsample_16_32_l = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_l')
                if l_last_res_planes == 16:
                    self.downsample_16_32_ll = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_ll')
            if self.planes == 64:
                if self.in_planes == 16:
                    self.downsample_16_64_x = Downsample_clean(16, 64, 1)
                    # print('downsample_16_32_x')
                if self.last_res_planes == 16:
                    self.downsample_16_64_l = Downsample_clean(16, 64, 1)
                    # print('downsample_16_32_l')
                if self.l_last_res_planes == 16:
                    self.downsample_16_64_ll = Downsample_clean(16, 64, 1)
                if self.in_planes == 32:
                    self.downsample_32_64_x = Downsample_clean(32, 64, 2)
                if self.last_res_planes == 32:
                    self.downsample_32_64_l = Downsample_clean(32, 64, 2)
                if self.l_last_res_planes == 32:
                    self.downsample_32_64_ll = Downsample_clean(32, 64, 2)
            if self.planes == 128:
                if self.in_planes == 64:
                    self.downsample_64_128_x = Downsample_clean(64, 128, 2)
                if self.last_res_planes == 64:
                    self.downsample_64_128_l = Downsample_clean(64, 128, 2)
                if self.l_last_res_planes == 64:
                    self.downsample_64_128_ll = Downsample_clean(64, 128, 2)
            if self.planes == 256:
                if self.in_planes == 128:
                    self.downsample_128_256_x = Downsample_clean(128, 256, 2)
                if self.last_res_planes == 128:
                    self.downsample_128_256_l = Downsample_clean(128, 256, 2)
                if self.l_last_res_planes == 128:
                    self.downsample_128_256_ll = Downsample_clean(128, 256, 2)

    def forward(self, x, last_res, l_last_res):
        # if self.expansion==4:
        #     residual = self.downsample_16_64_res(x)
        # elif self.expansion==1:
        #     residual = x
        residual = x
        F_x_n = self.bn1(x)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv1(F_x_n)

        F_x_n = self.bn2(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)

        F_x_n = self.bn3(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv3(F_x_n)

        # self.planes = self.planes*self.expansion

        # if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
        #     print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()',  F_x_n.size()[1], residual.size()[1],last_res.size()[1],l_last_res.size()[1])
        #     print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)
        # elif not (isinstance(last_res,int)):
        #     print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()', F_x_n.size()[
        #     1], residual.size()[1], last_res.size()[1], l_last_res)
        #     print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)
        # else:
        #     print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()',  F_x_n.size()[1], residual.size()[1],last_res,l_last_res)
        #     print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)
        if not (isinstance(last_res, int) or isinstance(l_last_res, int)):
            # print('HO')
            # if 1:
            if self.planes == 32:
                if self.in_planes == 16:
                    residual = self.downsample_16_32_x(residual)
                    # print('residual.size()', residual.size())
                if self.last_res_planes == 16:
                    last_res = self.downsample_16_32_l(last_res)
                # print('last_res.size()', last_res.size())
                if self.l_last_res_planes == 16:
                    l_last_res = self.downsample_16_32_ll(l_last_res)
                    # print('l_last_res.size()', l_last_res.size())
            if self.planes == 64:
                if self.in_planes == 16:
                    residual = self.downsample_16_64_x(residual)
                if self.last_res_planes == 16:
                    last_res = self.downsample_16_64_l(last_res)
                if self.l_last_res_planes == 16:
                    l_last_res = self.downsample_16_64_ll(l_last_res)
                if self.in_planes == 32:
                    residual = self.downsample_32_64_x(residual)
                if self.last_res_planes == 32:
                    last_res = self.downsample_32_64_l(last_res)
                if self.l_last_res_planes == 32:
                    l_last_res = self.downsample_32_64_ll(l_last_res)

            if self.planes == 128:
                if self.in_planes == 64:
                    residual = self.downsample_64_128_x(residual)
                if self.last_res_planes == 64:
                    last_res = self.downsample_64_128_l(last_res)
                if self.l_last_res_planes == 64:
                    l_last_res = self.downsample_64_128_ll(l_last_res)
            if self.planes == 256:
                if self.in_planes == 128:
                    residual = self.downsample_128_256_x(residual)
                if self.last_res_planes == 128:
                    last_res = self.downsample_128_256_l(last_res)
                if self.l_last_res_planes == 128:
                    l_last_res = self.downsample_128_256_ll(l_last_res)
            if not (isinstance(last_res, int) or isinstance(l_last_res, int)):
                if not self.fix_k:
                    self.b_0 = (3 * self.k - 1) / (self.k * 2)
                    self.a_0 = (3 * self.k + 3) / (self.k * 4)
                    self.a_1 = -1 / (self.k)
                    self.a_2 = (self.k + 1) / (4 * self.k)
                # x = torch.mul(b_0, F_x_n) + torch.mul(a_0, residual) + torch.mul(a_1, last_res) + torch.mul(a_2, l_last_res)

                x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(
                    self.a_1, last_res) + torch.mul(self.a_2, l_last_res)

        else:
            # print('res')
            x = F_x_n
        l_last_res = last_res
        last_res = residual  # x means the residual
        # residual = x
        # print('x.sixe()[1], residual.size()[1]', x.size()[1], residual.size()[1])
        return x, last_res, l_last_res, self.k


class Downsample(nn.Module):  # ReLU and BN are involved in this downsample
    def __init__(self, in_planes, out_planes, stride=2):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


class Downsample_clean(nn.Module):  # ReLU and BN are involved in this downsample
    def __init__(self, in_planes, out_planes, stride=2):
        super(Downsample_clean, self).__init__()
        self.downsample_ = nn.Sequential(
            # nn.BatchNorm2d(in_planes),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x):
        x = self.downsample_(x)
        return x


class Downsample_real(nn.Module):  # ReLU and BN are not involved in this downsample
    def __init__(self, in_shape, out_shape):
        super(Downsample_real, self).__init__()
        # in_shape = x.shape()
        self.in_planes = in_shape[1]
        self.out_planes = out_shape[1]
        self.stride = int(in_shape[2] / out_shape[2])
        # [256, 64, 32, 32]->[256, 128, 16, 16]
        self.downsample_real = nn.Sequential(
            # nn.BatchNorm2d(in_planes),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.in_planes, self.out_planes,
                      kernel_size=1, stride=self.stride, bias=False)
        )

    def forward(self, x):
        x = self.downsample_real(x)
        return x


class MResNet(nn.Module):

    # def __init__(self,block,layers,pretrain=True,num_classes=num_cla,stochastic_depth=False,PL=0.5,noise_level=0.001,noise=False):
    def __init__(self, block, layers, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0,
                 noise_level=0.001, noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super(MResNet, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.ks = nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(1.0, 1.1)) for i in
                                    range(layers[0] + layers[1] + layers[2])])  # each layer has a trainable $k_n$
        self.stochastic_depth = stochastic_depth
        blocks = []
        n = layers[0] + layers[1] + layers[2]

        if not self.stochastic_depth:
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
                self.in_planes = self.planes[i] * block.expansion
                for j in range(1, layers[
                    i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 0 to 2
                    blocks.append(block(self.in_planes, self.planes[i]))  # three (Basic) Blocks
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.strides[i],
                                    death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.in_planes = self.planes[i] * block.expansion
                for j in range(1, layers[i]):
                    blocks.append(block(self.in_planes, self.planes[i], death_rate=death_rates[i * layers[0] + j]))
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride=2):
        # self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21 = Downsample(16 * block.expansion,
                                       32 * block.expansion)  # "expansion" is 1 for BasicBlocks and is 4 for the Bottleneck
        # self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31 = Downsample(32 * block.expansion, 64 * block.expansion)
        # self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample

            residual = self.downsample1(x)  # residual.size()[1]: 16->64
        else:
            residual = x
        x = self.blocks[0](x) + residual  # x.size()[1]: 16->64
        last_res = residual
        for i, b in enumerate(self.blocks):  # index and content
            if i == 0:
                continue
            residual = x

            if b.in_planes != b.planes * b.expansion:  # sizes of the input and output are not the same
                if b.planes == 32:
                    residual = self.downsample21(x)
                    # if not self.pretrain:
                    # last_res=self.downsample22(last_res)
                elif b.planes == 64:
                    residual = self.downsample31(x)
                    # if not self.pretrain:
                    # last_res=self.downsample32(last_res)
                x = b(x)
                # print(x.size())
                # print(residual.size())
                x += residual

            elif self.pretrain:  #
                x = b(x) + residual
            else:  # in.channel = out.channel and not pretrain
                x = b(x) + self.ks[i].expand_as(residual) * residual + (1 - self.ks[i]).expand_as(
                    last_res) * last_res  # "B.expand_as (A)": expand B in A's shape

            last_res = residual

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.ks




class HONet_v2(nn.Module):

    def __init__(self, block, layers, k_ini=-9.0 / 5, pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(HONet_v2, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.ks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):  # there are 3 elements in the list like [7,7,7]
                # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini))
                # ###
                # if
                #
                # ###
                # self.l_last_res_planes = self.last_res_planes
                # self.last_res_planes = self.in_planes
                if l == 0 or l == 1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                # print('l', l)
                # print('i', i)
                for j in range(1, layers[
                    i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 1 to 2
                    # if l == 0:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    #
                    # elif l==1:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    # else:
                    # self.l_last_res_planes = self.planes[i]*block.expansion
                    # self.last_res_planes = self.planes[i]*block.expansion
                    # self.plane = self.planes[i]*block.expansion
                    # print('j', j)
                    # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini))  # three (Basic) Blocks
                    # self.l_last_res_planes = self.last_res_planes
                    # self.last_res_planes = self.in_planes
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1
                    # print('l', l)
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                # print('i', i)
                for j in range(1, layers[i]):
                    # print('j', j)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, death_rate=death_rates[i * layers[0] + j]))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride):

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        self.ks = []
        x = self.conv1(x)
        last_res = -1
        l_last_res = -1
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
            residual = self.downsample1(x)
            # print('downsample1')
        else:
            residual = x

        x, last_res, l_last_res, k = self.blocks[0](x, last_res, l_last_res)
        # print('v2: x.sixe()[1], residual.size()[1]', x.size()[1], residual.size()[1])
        x += residual
        # l_last_res = residual
        residual = x

        x, last_res, l_last_res, k = self.blocks[1](x, last_res, l_last_res)
        # x = self.blocks[1](x)[0] + residual
        x += residual
        # last_res = residual
        # residual = x # moved from below. Flag:318
        ### \end

        for i, b in enumerate(self.blocks):  # index and content
            if i == 0 or i == 1:
                # print('i', i)
                continue
            residual = x  # moved up. Flag:318
            ####
            # if b.in_planes != b.planes * b.expansion:  # sizes of the input and output are not the same
            #     if b.planes == 32:
            #         residual = self.downsample21(x)
            #         # if not self.pretrain:
            #         # last_res=self.downsample22(last_res)
            #     elif b.planes == 64:
            #         residual = self.downsample31(x)
            #
            #     x = b(x)
            #     # print(x.size())
            #     # print(residual.size())
            #     x += residual
            ####
            if self.pretrain:  #
                x = b(x) + residual

            else:  # in.channel = out.channel and not pretrain
                # \begin HONet core

                x, last_res, l_last_res, k = b(x, last_res, l_last_res)

                self.ks += k.data
                # print('i, ks', i, self.ks)

                # \end HONet core
            # print('cnt', cnt1, cnt2, cnt3, cnt4)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print('out')
        return x, self.ks


class HONet_stepsize(nn.Module):

    def __init__(self, block, layers, k_ini=-9.0 / 5, pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False, dataset='cifar10'):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(HONet_stepsize, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        self.stepsize = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.ks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):  # there are 3 elements in the list like [7,7,7]
                # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, stepsize=self.stepsize))
                # ###
                # if
                #
                # ###
                # self.l_last_res_planes = self.last_res_planes
                # self.last_res_planes = self.in_planes
                if l == 0 or l == 1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                # print('l', l)
                # print('i', i)
                for j in range(1, layers[
                    i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 1 to 2
                    # if l == 0:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    #
                    # elif l==1:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    # else:
                    # self.l_last_res_planes = self.planes[i]*block.expansion
                    # self.last_res_planes = self.planes[i]*block.expansion
                    # self.plane = self.planes[i]*block.expansion
                    # print('j', j)
                    # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, stepsize=self.stepsize))  # three (Basic) Blocks
                    # self.l_last_res_planes = self.last_res_planes
                    # self.last_res_planes = self.in_planes
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1
                    # print('l', l)
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, stepsize=self.stepsize,
                          death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                # print('i', i)
                for j in range(1, layers[i]):
                    # print('j', j)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, stepsize=self.stepsize,
                                        death_rate=death_rates[i * layers[0] + j]))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride):

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
    