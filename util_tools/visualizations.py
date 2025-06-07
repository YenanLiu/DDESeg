# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import os 
import torch
import numpy as np

from .colors import avss_pallete, vpo_pallate
from PIL import Image

def colored_masks(mask, task):
    if "VPO" in task:
        pallete = vpo_pallate
    else:
        pallete = avss_pallete

    valid_num, H, W = mask.shape
    rgb_mask = np.zeros((valid_num, H, W, 3), dtype=np.uint8)

    for cls_idx in range(len(pallete)):
        rgb = pallete[cls_idx]
        rgb_mask[mask == cls_idx] = rgb
    return rgb_mask

def mask_vis(mask, img_paths, task, save_dir=None):
    if "VPO" in task:
        pallete = vpo_pallate
    else:
        pallete = avss_pallete
        
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    B, H, W, = mask.shape

    rgb_mask = np.zeros((B, mask.shape[1], mask.shape[2], 3), np.uint8) # [B, H, W, 3]
    for cls_idx in range(0, len(pallete)):
        rgb = pallete[cls_idx]
        rgb_mask[mask == cls_idx] = rgb

    blended_imgs = []
    for b in range(B):
        img_path = img_paths[b]
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((W, H), Image.ANTIALIAS)
        rgb_m = Image.fromarray(rgb_mask[b]).convert("RGB")
        blend_img = Image.blend(rgb_m, img_resized, alpha=0.6)
        blended_imgs.append(np.asarray(blend_img))
        if save_dir is not None:
            v_name = img_path.split("/")[-3]
            img_index = os.path.basename(img_path).split(".")[0]
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, v_name + "_" + img_index + ".png")
            
    return blended_imgs



























