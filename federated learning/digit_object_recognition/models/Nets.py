#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x_out = F.relu(self.fc1(x))
        x = self.fc2(x_out)
        return x, x_out


class CNNCifarStd5(nn.Module):
    """659,818: """
    def __init__(self):
        super(CNNCifarStd5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 64 * 4 * 4)
        x_out = F.relu(self.fc1(x))
        x = self.fc2(x_out)
        return x, x_out


class CNNEmnistStd5(nn.Module):
    """438,074: """
    def __init__(self):
        super(CNNEmnistStd5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=0, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 64 * 3 * 3)
        x_out = F.relu(self.fc1(x))
        x = self.fc2(x_out)
        return x, x_out


class CNNCifar100Std5(nn.Module):
    """705,988: """
    def __init__(self):
        super(CNNCifar100Std5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 64 * 4 * 4)
        x_out = F.relu(self.fc1(x))
        x = self.fc2(x_out)
        return x, x_out




