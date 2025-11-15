from math import ceil
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import torch.utils.model_zoo as model_zoo
import kmeans1d
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten


class SAN(nn.Module):

    def __init__(self, inplanes, selected_classes=None, affine_par=True, resnet=False):
        super(SAN, self).__init__()
        self.margin = 0
        self.IN = nn.InstanceNorm2d(inplanes, affine=affine_par)
        self.selected_classes = selected_classes
        self.CFR_branches = nn.ModuleList()
        self.IN_branches = nn.ModuleList()
        for i in selected_classes:
            self.CFR_branches.append(
                nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False))
            self.IN_branches.append(
                nn.InstanceNorm2d(inplanes, affine=affine_par)
            )
        self.resnet = resnet
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.mask_matrix = None
        self.resnet = resnet

    def forward(self, x, masks):
        outs = []
        idx = 0
        masks = F.softmax(masks, dim=1)
        for index, i in enumerate(self.selected_classes):
            mask = torch.unsqueeze(masks[:, i, :, :], 1)
            mid = x * mask
            avg_out = torch.mean(mid, dim=1, keepdim=True)
            max_out, _ = torch.max(mid, dim=1, keepdim=True)
            atten = torch.cat([avg_out, max_out, mask], dim=1)
            atten = self.sigmoid(self.CFR_branches[index](atten))
            out = mid * atten
            out = self.IN_branches[index](out)
            outs.append(out)
        out_ = sum(outs)

        if self.resnet:
            out_ = out_ + x
        out_ = self.relu(out_)

        return out_
