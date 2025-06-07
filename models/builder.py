# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

from .audio_branch import HTSAT_Swin_Transformer
from .backbones import FAVSBackbone, SwinEnhancerTransformer

def build_audio_model(cfg):
    model = HTSAT_Swin_Transformer(
        spec_size=cfg.spec_size,
        patch_size=cfg.patch_size,
        patch_stride=cfg.patch_stride,
        in_chans=cfg.in_chans, 
        num_classes=cfg.num_classes,
        embed_dim=cfg.embed_dim, 
        depths=cfg.depths,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        config=cfg.config
    )
    return model


def build_favs_model(config):
    model = FAVSBackbone(
        in_channels=config.in_channels, 
        num_heads=config.num_heads, 
        mlp_ratios=config.mlp_ratios, 
        qkv_bias=config.qkv_bias, 
        qk_scale=config.qk_scale,
        drop_rate=config.drop_rate, 
        attn_drop_rate=config.attn_drop_rate,
        drop_path_rate=config.drop_path_rate, 
        depths=config.depths, 
        sr_ratios=config.sr_ratios
    )
    return model

def build_swin_enhancer_model(config):
    model = SwinEnhancerTransformer(
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        window_size=config.window_size,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        qk_scale=config.qk_scale,
        drop_rate=config.drop_rate,
        attn_drop_rate=config.attn_drop_rate,
        drop_path_rate=config.drop_path_rate,
        ape=config.ape,
        patch_norm=config.patch_norm,
        use_checkpoint=config.use_checkpoint,
    )
    return model