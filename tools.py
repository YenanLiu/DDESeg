# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import torch
import numpy as np
import os
import cv2
import torch.distributed as dist

from PIL import Image, ImageFilter, ImageChops
from util_tools import *

def safe_all_reduce(tensor, op=dist.ReduceOp.SUM, fallback=None, fatal=True):
    rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        dist.all_reduce(tensor, op=op)
        return tensor
    except Exception as e:
        print(f"[Rank {rank}] NCCL all_reduce failed: {e}")
        if fatal:
            if dist.is_initialized():
                dist.destroy_process_group()
            raise e
        else:
            return fallback if fallback is not None else tensor
        
def reduce_tensor(tensor, world_size=None):
    if world_size is None:
        world_size = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def normalize_sim_map_converting(sim_map):
    """
    Args:
        sim_map: (np.ndarray) [H, W] 
    Returns:
        normalized_map (np.ndarray): the value range is in [0, 255]
    """
    norm_sim_map = cv2.normalize(sim_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm_sim_map.astype(np.uint8)

def simmap_to_img(sim_map, colormap=cv2.COLORMAP_JET):
    if isinstance(sim_map, torch.Tensor):
        sim_map = sim_map.detach().cpu().numpy() 

    norm_sim_map = normalize_sim_map_converting(sim_map)
    color_image = cv2.applyColorMap(norm_sim_map, colormap)
    pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    return pil_image

@torch.no_grad()
def vis_masks(pred_masks, gt_masks, frame_paths, task=None, vis_dict=None, vis_save_dir=None):
    """_summary_

    Args:
        pred_masks (_type_): [valid_num, C, H, W]
        gt_masks (_type_): [valid_num, H, W]
        frame_paths (_type_): _description_
        vis_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert len(frame_paths) == pred_masks.shape[0], "the valid data amount are inconsistent between the frame_paths and pred_masks!!!"
    assert len(frame_paths) == gt_masks.shape[0], "the valid data amount are inconsistent between the frame_paths and gt masks!!!"

    wandb = get_wandb()
 
    _, H, W = gt_masks.shape
    pred_masks = pred_masks.cpu().numpy()
    gt_masks = gt_masks.cpu().numpy()
    
    pred_rgb_masks = colored_masks(pred_masks, task) # [valid_num, H, W, 3]
    gt_rgb_masks = colored_masks(gt_masks, task) # [valid_num, H, W, 3]

    for pred_mask, gt_mask, frame_path in zip(pred_rgb_masks, gt_rgb_masks, frame_paths):
        img = Image.open(frame_path).convert('RGB')
        img_resized = img.resize((W, H), Image.LANCZOS)

        pred_mask_img = Image.fromarray(pred_mask)
        gt_mask_img = Image.fromarray(gt_mask)

        blend_pred_img = Image.blend(pred_mask_img, img_resized, alpha=0.3)
        blend_gt_img = Image.blend(gt_mask_img, img_resized, alpha=0.3)
        
        combined_width = blend_pred_img.width + blend_gt_img.width
        combined_img = Image.new('RGB', (combined_width, blend_pred_img.height))
        combined_img.paste(blend_pred_img, (0, 0))  #  pred_mask blend on the left
        combined_img.paste(blend_gt_img, (blend_pred_img.width, 0))  # gt_mask blend on the right

        v_name = frame_path.split("/")[-3]
        img_index = os.path.basename(frame_path).split(".")[0]
        uid = v_name + "_" + img_index

        if vis_save_dir is not None:
            os.makedirs(vis_save_dir, exist_ok=True)
            vis_save_path = os.path.join(vis_save_dir, uid + ".png")
            combined_img = combined_img.convert('RGB')
            combined_img.save(vis_save_path)

        if vis_dict is not None and wandb is not None:
            img_list = vis_dict.setdefault(uid, [])
            img_list.append(wandb.Image(combined_img))
    
    return vis_dict


@torch.no_grad()
def vis_eval_masks(pred_masks, gt_masks, frame_paths, task=None, vis_save_dir=None, pred_masks_prob=None):
    """_summary_

    Args:
        pred_masks (_type_): [valid_num, C, H, W]
        gt_masks (_type_): [valid_num, H, W]
        frame_paths (_type_): _description_
        vis_dict (_type_): _description_
        pred_masks_prob: [valid_num, C, H, W]

    Returns:
        _type_: _description_
    """
    assert len(frame_paths) == pred_masks.shape[0], "the valid data amount are inconsistent between the frame_paths and pred_masks!!!"
    assert len(frame_paths) == gt_masks.shape[0], "the valid data amount are inconsistent between the frame_paths and gt masks!!!"

    wandb = get_wandb()
 
    B, H, W = gt_masks.shape
    pred_masks = pred_masks.cpu().numpy()
    gt_masks = gt_masks.cpu().numpy()

    if pred_masks_prob is not None:
        pred_masks_prob_avg = pred_masks_prob.mean(dim=1)
        pred_masks_prob_array = pred_masks_prob_avg.detach().cpu().numpy()
    
    pred_rgb_masks = colored_masks(pred_masks, task) # [valid_num, H, W, 3]
    gt_rgb_masks = colored_masks(gt_masks, task) # [valid_num, H, W, 3]

    for pred_mask, gt_mask, frame_path in zip(pred_rgb_masks, gt_rgb_masks, frame_paths):
        img = Image.open(frame_path).convert('RGB')
        img_resized = img.resize((W, H), Image.LANCZOS)

        pred_mask_img = Image.fromarray(pred_mask)
        gt_mask_img = Image.fromarray(gt_mask)

        blend_pred_img = Image.blend(pred_mask_img, img_resized, alpha=0.4)
        binary_mask = (pred_mask > 0).astype(np.uint8) * 255
        binary_mask_img = Image.fromarray(binary_mask).convert('L')
        dilated_mask = binary_mask_img.filter(ImageFilter.MaxFilter(3))
        edges = ImageChops.difference(dilated_mask, binary_mask_img)

        edges_rgb = edges.convert('RGB')
        edges_np = np.array(edges_rgb)
        edges_np[edges_np != 0] = 255
        edges_img = Image.fromarray(edges_np)

        blend_pred_img.paste(edges_img, mask=edges) 
    
        v_name = frame_path.split("/")[-3]
        img_index = os.path.basename(frame_path).split(".")[0]
        uid = v_name + "_" + img_index
        blend_pred_img = blend_pred_img.convert('RGB')

        os.makedirs(vis_save_dir, exist_ok=True)
        if 'WEdge' not in vis_save_dir:
            new_vis_save_dir = os.path.join(vis_save_dir, 'WEdge')
            os.makedirs(new_vis_save_dir, exist_ok=True)
        else:
            new_vis_save_dir = vis_save_dir
        vis_save_path = os.path.join(new_vis_save_dir, uid + ".png")
        blend_pred_img.save(vis_save_path)
        print(vis_save_path)

        new_dir = new_vis_save_dir.replace('WEdge', 'WOEdge')
        os.makedirs(new_dir, exist_ok=True)
        wo_vis_save_path = os.path.join(new_dir, uid + ".png")
        wo_blend_pred_img = Image.blend(pred_mask_img, img_resized, alpha=0.4)
        wo_blend_pred_img.save(wo_vis_save_path)
        print(wo_vis_save_path)
 

def valid_mask(effective_mask, mask):
    """_summary_

    Args:
        effective_mask (_type_): [B, T]
        mask (_type_): [B, T, H, W]
    return 
        valid_mask: [valid_num, H, W]
    """
    if len(mask.shape) == 4:
        B, T, H, W = mask.shape
        mask_flattened = mask.view(B*T, H, W)
    else:
        B, T, C, H, W = mask.shape
        mask_flattened = mask.view(B*T, C, H, W)

    effective_mask_flattened = effective_mask.view(-1) # [B*T]
    valid_indices = torch.nonzero(effective_mask_flattened).squeeze(dim=-1)

    valid_mask = mask_flattened[valid_indices] # [valid_num, H, w] or [valid_num, C, H, W]
    
    return valid_mask

def valid_path(effective_mask, paths):
    valid = torch.tensor(effective_mask, dtype=torch.bool).cpu()
    paths_matrix = np.array(paths)
    paths_matrix_transposed = paths_matrix.T
    valid_paths = paths_matrix_transposed[valid].tolist()
    return valid_paths