# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from shakedrop import ShakeDrop


class ShakeBasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, p_shakedrop=1.0):
        super(ShakeBasicBlock, self).__init__()
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_ch, out_ch, stride=stride)
        self.shortcut = not self.downsampled and None or nn.AvgPool2d(2)
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        h = self.branch(x)
        h = self.shake_drop(h)
        h0 = x if not self.downsampled else self.shortcut(x)
        pad_zero = Variable(torch.zeros(h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3)).float()).cuda()
        h0 = torch.cat([h0, pad_zero], dim=1)

        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakePyramidNet(nn.Module):

    def __init__(self, depth=110, alpha=270, label=10):
        super(ShakePyramidNet, self).__init__()
        in_ch = 16
        # for BasicBlock
        n_units = (depth - 2) // 6
        in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        block = ShakeBasicBlock

        self.in_chs, self.u_idx = in_chs, 0
        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        
        self.ll = nn.LeakyReLU(0.1)
        self.conv0 = nn.Conv2d(3, 4*4*16, kernel_size=4, stride=4, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(4*4*16)
        self.conv1 = nn.Conv2d(4*4*3, 4*4*3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(3*16)
        self.conv2 = nn.Conv2d(4*4*3, 16*16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(16*16)
        self.pixelshuffle = nn.PixelShuffle(4)        

        self.c_in = nn.Conv2d(16, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        self.layer1 = self._make_layer(n_units, block, 1)
        self.layer2 = self._make_layer(n_units, block, 2)
        self.layer3 = self._make_layer(n_units, block, 2)
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], label)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x_stack = None
        idx = 0
        for i in range(8):
          tmp = None
          for j in range(8):
            out = self.ll(self.bn0(self.conv0(x[:,:,i*4:(i+1)*4,j*4:(j+1)*4])))
            out = self.pixelshuffle(out)
            if tmp is None:
              tmp = out
            else:
              tmp = torch.cat([tmp,out],dim=3)
            idx = idx + 1
          if x_stack is None:
            x_stack = tmp
          else:
            x_stack = torch.cat([x_stack,tmp],dim=2)
        h = x_stack
        feature = h
        h = self.bn_in(self.c_in(h))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(self.bn_out(h))
        h = F.avg_pool2d(h, 8)
        h = h.view(h.size(0), -1)
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, block, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(block(self.in_chs[self.u_idx], self.in_chs[self.u_idx+1],
                                stride, self.ps_shakedrop[self.u_idx]))
            self.u_idx, stride = self.u_idx + 1, 1
        return nn.Sequential(*layers)

