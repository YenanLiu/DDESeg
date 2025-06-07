# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
#
# Modified by Chen Liu
# ---------------------------------------------------------------

import torch
import torch.nn as nn

from . import Block, OverlapPatchEmbed
from .av_malign import *
from .pre_modules import aud_Mlp


class FAVSBackbone(nn.Module):
    def __init__(self, *,
            in_channels=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[4, 4, 4, 4], 
            qkv_bias=True, 
            qk_scale=None,
            drop_rate=0.0, 
            attn_drop_rate=0.,
            drop_path_rate=0.1, 
            depths=[2, 2, 2, 2], 
            sr_ratios=[8, 4, 2, 1]):
        
        super(FAVSBackbone, self).__init__()

        norm_layer = nn.LayerNorm

        self.out_indices = len(depths)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=2, in_chans=3, embed_dim=in_channels[0])  
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=in_channels[0], embed_dim=in_channels[1]) 
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=in_channels[1], embed_dim=in_channels[2]) 
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=in_channels[2], embed_dim=in_channels[3])  
        
        # aud projection
        self.aud_projs = nn.ModuleList([
            aud_Mlp(in_channels=768, out_channels=out_channels, drop=0.0)
            for out_channels in in_channels
        ])

        self.av_aligners = nn.ModuleList([
            AV_Aligner(dim=out_channels, num_heads=8, num_clusters=5)
            for out_channels in in_channels
        ])

        # Initialize blocks for each stage
        self.block1 = nn.ModuleList(
            [Block(
                dim=in_channels[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]
            ) for i in range(depths[0])]
        )
        self.norm1 = norm_layer(in_channels[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [Block(
                dim=in_channels[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]
            ) for i in range(depths[1])]
        )
        self.norm2 = norm_layer(in_channels[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [Block(
                dim=in_channels[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]
            ) for i in range(depths[2])]
        )
        self.norm3 = norm_layer(in_channels[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [Block(
                dim=in_channels[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]
            ) for i in range(depths[3])]
        )
        self.norm4 = norm_layer(in_channels[3])


    def forward(self, x, aud_token):
        B = x.size(0)
        out = {}
        for i in range(self.out_indices):
 
            patch_embed = getattr(self, f'patch_embed{i+1}') # [B, H*W, dim]
            
            aud_proj = self.aud_projs[i]

            if i > 0:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            x, H, W = patch_embed(x) 
            aud_token_proj = aud_proj(aud_token)

            av_aligner = self.av_aligners[i]
            x = av_aligner(aud_token_proj, x)

            block = getattr(self, f'block{i+1}')
            norm = getattr(self, f'norm{i+1}')

            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            
            out[f"fea_{i}"] = x 
            out[f"H_{i}"] = H
            out[f"W_{i}"] = W

        return out