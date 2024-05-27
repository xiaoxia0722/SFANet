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

