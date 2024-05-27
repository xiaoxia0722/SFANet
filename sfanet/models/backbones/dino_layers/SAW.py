import time

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SAW(nn.Module):
    def __init__(self, selected_classes, dim, relax_denom=0, work=False, channel_num=1024, topk=20):
        super(SAW, self).__init__()
        self.work = work
        self.selected_classes = selected_classes
        self.lens = len(selected_classes)
        self.C = self.lens
        self.channel_num = channel_num
        self.dim = dim
        self.topk = topk
        self.i = torch.eye(self.C, self.C).cuda()
        self.reversal_i = torch.ones(self.C, self.C).triu(diagonal=1).cuda()
        self.num_off_diagonal = torch.sum(self.reversal_i)
        if relax_denom == 0:
            print("Note relax_denom == 0!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom


    def get_mask_matrix(self):
        return self.i, self.reversal_i, self.margin, self.num_off_diagonal

    def get_covariance_matrix(self, x, eye=None):
        eps = 1e-5
        B, C, H, W = x.shape  # i-th feature size (B X C X H X W)
        HW = H * W
        if eye is None:
            eye = torch.eye(C).cuda()
        x = x.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        x_cor = torch.bmm(x, x.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW

        return x_cor, B

    def instance_whitening_loss(self, x, eye, mask_matrix, margin, num_remove_cov):
        x_cor, B = self.get_covariance_matrix(x, eye=eye)
        x_cor_masked = x_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(x_cor_masked), dim=(1, 2), keepdim=True) - margin  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss
    def sort_with_idx(self, x, idx,weights):
        b,c,_,_ = x.size()
        after_sort = torch.zeros_like(x)
        weights = F.sigmoid(weights)
        for i in range(b):

            for k in range(int(c / self.lens)):
                for j in range(self.lens):
                    wgh = weights[self.selected_classes[j]][idx[self.selected_classes[j]][k]]
                    # [0 1 2  3 4 5] [5, 6, 7, 8, 9]
                    after_sort[i][self.lens*k+j][:][:] = wgh * x[i][idx[self.selected_classes[j]][k]][:][:]

        return after_sort


