# !/bin/bash

from torch import nn
import torch

import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore=255):
        super().__init__()
        self.ignore = ignore

    def forward(self, output, target):
        lis = []
        for i in range(19):
            # non_zero_num = torch.nonzero(target).shape[0]
            # print(type(non_zero_num))
            gt = (target == i).float()  # B
            inter = torch.sum(gt, dim=(0, 1, 2)) # B

            total_num = torch.prod(torch.tensor(target.shape)).float()

            k = inter / total_num

            lis.append(1-k)
        # print(lis)

        scaled_weight = torch.stack(lis, dim=0)


        if type(output) == list:
            loss = F.cross_entropy(output[0], target, weight=scaled_weight, ignore_index=self.ignore)
            for i in range(1, len(output)):
                loss += F.cross_entropy(output[i], target, weight=scaled_weight, ignore_index=self.ignore)
        else:
            loss = F.cross_entropy(output, target, weight=scaled_weight, ignore_index=self.ignore)

        return loss