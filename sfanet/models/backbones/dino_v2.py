# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from mmseg.models.builder import BACKBONES
from mmengine.model import BaseModule
import torch.nn.functional as F
from .dino_layers import (
    Mlp,
    PatchEmbed,
    SwiGLUFFNFused,
    MemEffAttention,
    SAN,
    SAW,
    NestedTensorBlock as Block,
)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


@BACKBONES.register_module()
class DinoVisionTransformer(BaseModule):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=partial(Block, attn_class=MemEffAttention),
        ffn_layer="mlp",
        block_chunks=1,
        out_indices=[7, 11, 15, 23],
        san=False,
        saw=False,
        selected_classes=[0, 10, 2, 1, 8],
        san_resnet=False,
        num_classes=19,
        init_cfg=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            san: enable SAN for feature if True
            saw: enable SAS for feature if True
        """
        super().__init__(init_cfg)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.out_indices = out_indices
        self.san = san
        self.saw = saw
        self.san_resnet = san_resnet

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        
        self.san_classifier_0 = None
        self.san_classifier_1 = None
        self.san_classifier_2 = None
        self.san_classifier_3 = None
        self.san_stage_0 = None
        self.san_stage_1 = None
        self.san_stage_2 = None
        self.san_stage_3 = None
        self.saw_stage_0 = None
        self.saw_stage_1 = None
        self.saw_stage_2 = None
        self.saw_stage_3 = None


        if san or saw:
            self.san_classifier_0 = nn.Conv2d(embed_dim, num_classes, kernel_size=1, stride=1, bias=True)
            self.san_classifier_1 = nn.Conv2d(embed_dim, num_classes, kernel_size=1, stride=1, bias=True)
            self.san_classifier_2 = nn.Conv2d(embed_dim, num_classes, kernel_size=1, stride=1, bias=True)
            self.san_classifier_3 = nn.Conv2d(embed_dim, num_classes, kernel_size=1, stride=1, bias=True)
        # SAN
        if self.san:
            self.san_stage_0 = SAN(inplanes=embed_dim, selected_classes=selected_classes, resnet=san_resnet)
            self.san_stage_1 = SAN(inplanes=embed_dim, selected_classes=selected_classes, resnet=san_resnet)
            self.san_stage_2 = SAN(inplanes=embed_dim, selected_classes=selected_classes, resnet=san_resnet)
            self.san_stage_3 = SAN(inplanes=embed_dim, selected_classes=selected_classes, resnet=san_resnet)
        if self.saw:
            self.saw_stage_0 = SAW(selected_classes, dim=embed_dim, relax_denom=2.0, work=True)
            self.saw_stage_1 = SAW(selected_classes, dim=embed_dim, relax_denom=2.0, work=True)
            self.saw_stage_2 = SAW(selected_classes, dim=embed_dim, relax_denom=2.0, work=True)
            self.saw_stage_3 = SAW(selected_classes, dim=embed_dim, relax_denom=2.0, work=True)

        self.train_mode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.1)
                # m.weight.data.normal_(0, 0.2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.init_weights()

    # def init_weights(self):
    #     trunc_normal_(self.pos_embed, std=0.02)
    #     nn.init.normal_(self.cls_token, std=1e-6)
    #     named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :]
                    .permute(0, 2, 1)
                    .reshape(B, -1, h // self.patch_size, w // self.patch_size)
                    .contiguous()
                )
        return outs

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        saw_loss_lay0 = torch.zeros(1)
        saw_loss_lay1 = torch.zeros(1)
        saw_loss_lay2 = torch.zeros(1)
        saw_loss_lay3 = torch.zeros(1)
        ret0 = None
        ret1 = None
        ret2 = None
        ret3 = None
        ret_0 = None
        ret_1 = None
        ret_2 = None
        ret_3 = None
        ret_cp = []
        if isinstance(ret[0], torch.Tensor):
            for c in ret:
                ret_cp.append(c)
            if self.san:
                ret0 = self.san_classifier_0(ret[0].detach())
                ret_0 = self.san_stage_0(ret[0], ret0)
                ret[0] = ret_0

                ret1 = self.san_classifier_1(ret[1].detach())
                ret_1 = self.san_stage_1(ret[1], ret1)
                ret[1] = ret_1

                ret2 = self.san_classifier_2(ret[2].detach())
                ret_2 = self.san_stage_2(ret[2], ret2)
                ret[2] = ret_2

                ret3 = self.san_classifier_3(ret[3].detach())
                ret_3 = self.san_stage_3(ret[3], ret3)
                ret[3] = ret_3

            if self.saw and self.train_mode:
                saw_loss_lay0 = self.saw_stage_0(ret[0], self.san_classifier_0)
                saw_loss_lay1 = self.saw_stage_1(ret[1], self.san_classifier_1)
                saw_loss_lay2 = self.saw_stage_2(ret[2], self.san_classifier_2)
                saw_loss_lay3 = self.saw_stage_3(ret[3], self.san_classifier_3)
            ret[0] = F.interpolate(
                ret[0], scale_factor=4, mode="bilinear", align_corners=False
            )
            ret[1] = F.interpolate(
                ret[1], scale_factor=2, mode="bilinear", align_corners=False
            )
            ret[3] = F.interpolate(
                ret[3], scale_factor=0.5, mode="bilinear", align_corners=False
            )
        else:
            for c in ret[0]:
                ret_cp.append(c)
            if self.san:
                ret0 = self.san_classifier_0(ret[0][0].detach())
                ret_0 = self.san_stage_0(ret[0][0], ret0)
                ret[0][0] = ret_0


                ret1 = self.san_classifier_1(ret[0][1].detach())
                ret_1 = self.san_stage_1(ret[0][1], ret1)
                ret[0][1] = ret_1

                ret2 = self.san_classifier_2(ret[0][2].detach())
                ret_2 = self.san_stage_2(ret[0][2], ret2)
                ret[0][2] = ret_2

                ret3 = self.san_classifier_3(ret[0][3].detach())
                ret_3 = self.san_stage_3(ret[0][3], ret3)
                ret[0][3] = ret_3

            if self.saw and self.train_mode:
                saw_loss_lay0 = self.saw_stage_0(ret[0][0], self.san_classifier_0)
                saw_loss_lay1 = self.saw_stage_1(ret[0][1], self.san_classifier_1)
                saw_loss_lay2 = self.saw_stage_2(ret[0][2], self.san_classifier_2)
                saw_loss_lay3 = self.saw_stage_3(ret[0][3], self.san_classifier_3)

            ret[0][0] = F.interpolate(
                ret[0][0], scale_factor=4, mode="bilinear", align_corners=False
            )
            ret[0][1] = F.interpolate(
                ret[0][1], scale_factor=2, mode="bilinear", align_corners=False
            )
            ret[0][3] = F.interpolate(
                ret[0][3], scale_factor=0.5, mode="bilinear", align_corners=False
            )

            ret_cp[0] = F.interpolate(
                ret_cp[0], scale_factor=4, mode="bilinear", align_corners=False
            )
            ret_cp[1] = F.interpolate(
                ret_cp[1], scale_factor=2, mode="bilinear", align_corners=False
            )
            ret_cp[3] = F.interpolate(
                ret_cp[3], scale_factor=0.5, mode="bilinear", align_corners=False
            )
        # return ret, [saw_loss_lay0, saw_loss_lay1, saw_loss_lay2, saw_loss_lay3], ret_cp, [self.san_stage_0, self.san_stage_1, self.san_stage_2, self.san_stage_3], [ret0, ret1, ret2, ret3]
        
        return ret, [saw_loss_lay0, saw_loss_lay1, saw_loss_lay2, saw_loss_lay3], ret_cp, [self.san_stage_0, self.san_stage_1, self.san_stage_2, self.san_stage_3], [ret_0, ret_1, ret_2, ret_3]
