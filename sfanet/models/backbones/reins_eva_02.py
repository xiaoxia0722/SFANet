from .dino_layers import SAN, SAW
from .eva_02 import EVA2
from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch import nn
from .utils import set_requires_grad, set_train


@BACKBONES.register_module()
class ReinsEVA2(EVA2):
    def __init__(self, reins_config=None, san=True, saw=True, selected_classes=[0, 10, 2, 1, 8], san_resnet=False, num_classes=19,**kwargs):
        super().__init__(**kwargs)
        self.reins: Reins = MODELS.build(reins_config)
        self.san = san
        self.saw = saw
        self.san_resnet = san_resnet
        self.num_classes = num_classes
        self.selected_classes = selected_classes

        if san or saw:
            self.san_classifier_0 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)
            self.san_classifier_1 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)
            self.san_classifier_2 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)
            self.san_classifier_3 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)
        # SAN
        if self.san:
            self.san_stage_0 = SAN(inplanes=1024, selected_classes=selected_classes, resnet=san_resnet)
            self.san_stage_1 = SAN(inplanes=1024, selected_classes=selected_classes, resnet=san_resnet)
            self.san_stage_2 = SAN(inplanes=1024, selected_classes=selected_classes, resnet=san_resnet)
            self.san_stage_3 = SAN(inplanes=1024, selected_classes=selected_classes, resnet=san_resnet)
        if self.saw:
            self.saw_stage_0 = SAW(selected_classes, dim=1024, relax_denom=2.0, work=True)
            self.saw_stage_1 = SAW(selected_classes, dim=1024, relax_denom=2.0, work=True)
            self.saw_stage_2 = SAW(selected_classes, dim=1024, relax_denom=2.0, work=True)
            self.saw_stage_3 = SAW(selected_classes, dim=1024, relax_denom=2.0, work=True)

        self.train_mode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
            x = self.reins.forward(
                x,
                i,
                batch_first=True,
                has_cls_token=True,
            )
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())

        saw_loss_lay0 = torch.zeros(1)
        saw_loss_lay1 = torch.zeros(1)
        saw_loss_lay2 = torch.zeros(1)
        saw_loss_lay3 = torch.zeros(1)
        ret_cp = []

        ret0 = None
        ret1 = None
        ret2 = None
        ret3 = None

        for c in features:
            ret_cp.append(c)
        if self.san:
            ret0 = self.san_classifier_0(features[0].detach())
            ret_0 = self.san_stage_0(features[0], ret0)
            if self.san_resnet:
                features[0] = ret_0 + features[0]
            else:
                features[0] = ret_0

            ret1 = self.san_classifier_1(features[1].detach())
            ret_1 = self.san_stage_1(features[1], ret1)
            if self.san_resnet:
                features[1] = ret_1 + features[1]
            else:
                features[1] = ret_1

            ret2 = self.san_classifier_2(features[2].detach())
            ret_2 = self.san_stage_2(features[2], ret2)
            if self.san_resnet:
                features[2] = ret_2 + features[2]
            else:
                features[2] = ret_2

            ret3 = self.san_classifier_3(features[3].detach())
            ret_3 = self.san_stage_3(features[3], ret3)
            if self.san_resnet:
                features[3] = ret_3 + features[3]
            else:
                features[3] = ret_3

        if self.saw and self.train_mode:
            saw_loss_lay0 = self.saw_stage_0(features[0], self.san_classifier_0)
            saw_loss_lay1 = self.saw_stage_1(features[1], self.san_classifier_1)
            saw_loss_lay2 = self.saw_stage_2(features[2], self.san_classifier_2)
            saw_loss_lay3 = self.saw_stage_3(features[3], self.san_classifier_3)

        features[0] = F.interpolate(
            features[0], scale_factor=4, mode="bilinear", align_corners=False
        )
        features[1] = F.interpolate(
            features[1], scale_factor=2, mode="bilinear", align_corners=False
        )
        features[3] = F.interpolate(
            features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        )
        return self.reins.return_auto(features), [saw_loss_lay0, saw_loss_lay1, saw_loss_lay2, saw_loss_lay3], ret_cp, [self.san_stage_0, self.san_stage_1, self.san_stage_2, self.san_stage_3], [ret0, ret1, ret2, ret3]

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", 'sfn'])
        set_train(self, ["reins", 'sfn'])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "rein" not in k and 'san' not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
