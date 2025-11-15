from mmpretrain.models.backbones import ConvNeXt
from mmseg.models.builder import BACKBONES, MODELS

from .dino_layers import SAN, SAW
from .reins import Reins
from .utils import set_requires_grad, set_train
from typing import List, Dict
import torch.nn as nn


@BACKBONES.register_module()
class ReinsConvNeXt(ConvNeXt):
    def __init__(
        self,
        distinct_cfgs: List[Dict] = None,
        reins_config: Dict = None,
        san=True,
        saw=True,
        san_resnet=False,
        num_classes=19,
        selected_classes=[16, 3, 9, 7, 18],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins: List[Reins] = nn.ModuleList()
        self.sfns = nn.ModuleList()
        self.sfws = nn.ModuleList()
        self.sfn_classifiers = nn.ModuleList()
        self.san = san
        self.saw = saw
        self.san_resnet = san_resnet
        self.selected_classes = selected_classes
        self.num_classes = num_classes
        for cfgs in distinct_cfgs:
            reins_config.update(cfgs)
            self.reins.append(MODELS.build(reins_config))
            if self.san:
                self.sfn_classifiers.append(nn.Conv2d(cfgs['embed_dims'], self.num_classes, kernel_size=1, stride=1, bias=True))
                self.sfns.append(SAN(inplanes=cfgs['embed_dims'], selected_classes=self.selected_classes, resnet=self.san_resnet))
                if self.saw:
                    self.sfws.append(SAW(selected_classes, dim=cfgs['embed_dims'], relax_denom=2.0, work=True))

        self.train_mode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            for idx_sublayer, sublayer in enumerate(stage):
                x = sublayer(x)
                B, C, H, W = x.shape
                x = (
                    self.reins[i]
                    .forward(
                        x.flatten(-2, -1).permute(0, 2, 1),
                        idx_sublayer,
                        batch_first=True,
                        has_cls_token=False,
                    )
                    .permute(0, 2, 1)
                    .reshape(B, C, H, W)
                )
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(self.reins[i].return_auto(norm_layer(gap).flatten(1)))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(self.reins[i].return_auto(norm_layer(x).contiguous()))

        out1, out2 = [f1 for f1, _ in outs], sum([f2 for _, f2 in outs])

        ret_cp = []
        saw_loss = []
        rets = []
        if self.san:
            for c in out1:
                ret_cp.append(c)
            for i in range(len(out1)):
                ret0 = self.sfn_classifiers[i](out1[i].detach())
                rets.append(ret0)
                ret_0 = self.sfns[i](out1[i], ret0)
                if self.san_resnet:
                    out1[i] = out1[i] + ret_0
                else:
                    out1[i] = ret_0
                if self.saw and self.train_mode:
                    saw_loss.append(self.sfws[i](out1[i], self.sfn_classifiers[i]))
        return (out1, out2), saw_loss, ret_cp, self.sfns, rets

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "sfn"])
        set_train(self, ["reins", "sfn"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "rein" not in k and 'san' not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
