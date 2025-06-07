# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

def CE_Loss(inputs, target, cls_weights=None):
    num_classes = inputs.shape[1]
    if num_classes == 1: 
        inputs = inputs.squeeze(1)  # [B, H, W]
        loss = nn.BCEWithLogitsLoss(weight=cls_weights)(inputs, target.float())
    else: 
        loss = nn.CrossEntropyLoss(weight=cls_weights)(inputs, target.long())
    return loss

def Focal_Loss(inputs, target, cls_weights=None, alpha=0.5, gamma=2):
    """
    Multiclass Focal Loss

    Args:
    inputs (torch.Tensor): [batch_size, num_classes, height, width]
    target (torch.Tensor): ground truth [batch_size, height, width]
    cls_weights (torch.Tensor): [num_classes] the weight for classes
    gamma (float): the focal weight for Focal Loss
    Returns:
        torch.Tensor: the focal loss
    """
    num_classes = inputs.shape[1]

    if num_classes == 1: 
        inputs = inputs.squeeze(1)  # [B, H, W]
        ce_loss = nn.BCEWithLogitsLoss(weight=cls_weights, reduction='none')(inputs, target.float())
    else:  
        ce_loss = nn.CrossEntropyLoss(weight=cls_weights, reduction='none')(inputs, target.long())

    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma) * ce_loss # [batch_size, height, width]

    if alpha is not None:
        if isinstance(alpha, (float, int)):
            alpha_t = torch.ones_like(target, dtype=torch.float32) * alpha
        elif isinstance(alpha, torch.Tensor) and alpha.numel() == num_classes:
            alpha_t = alpha[target] # [batch_size, height, width]
        else:
            raise ValueError("alpha should be a float int or torch.Tensor with [num_classes] shape")

        focal_loss = alpha_t * focal_loss # [batch_size, height, width]

    return focal_loss.mean()

def weighted_Dice_Loss(inputs, targets, smooth=1e-5, weight=None):
    # inputs (_type_): [valid_num, C, H, W]
    # targets (_type_):  [valid_num, H, W]
    
    num_classes = inputs.shape[1]
    
    if num_classes == 1:  
        inputs = torch.sigmoid(inputs)
        targets = targets.float()
    else:  
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    batch_size = inputs.shape[0]
    inputs = inputs.contiguous().view(batch_size, num_classes, -1)
    targets = targets.contiguous().view(batch_size, num_classes, -1)

    intersection = (inputs * targets).sum(2)  # [B, num_classes]
    union = inputs.sum(2) + targets.sum(2)  # [B, num_classes]

    dice = (2. * intersection + smooth) / (union +  smooth) # [B, num_classes]

    if weight is not None:
        dice = dice * weight.to(inputs.device)

    loss = 1 - dice.mean()
    return loss

def IoU_Loss(inputs, targets, smooth=1e-5, weight=None):
    """
    Args:
        inputs (torch.Tensor): Model predictions, shape [batch_size, num_classes, height, width].
        targets (torch.Tensor): Ground truth labels, shape [batch_size, height, width].
        smooth (float): A small value to avoid division by zero, defaults to 1e-6.

    Returns:
        torch.Tensor: Computed IoU loss.
    """
    num_classes = inputs.shape[1]

    if num_classes == 1: 
        inputs = torch.sigmoid(inputs)
        targets = targets.float()
    else:  
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    batch_size = inputs.shape[0]
    inputs = inputs.contiguous().view(batch_size, num_classes, -1)
    targets = targets.contiguous().view(batch_size, num_classes, -1)
        
    intersection = (inputs * targets).sum(2)
    union = inputs.sum(2) + targets.sum(2) - intersection
        
    iou = (intersection + smooth) / (union + smooth)

    if weight is not None:
        iou = iou * weight.to(inputs.device)

    iou_loss = 1 - iou.mean()

    return iou_loss

def loss_calculation(pred_mask, gt_mask, cfg, device):
    """_summary_
    Args:
        pred_mask (_type_): [valid_num, C, H, W]
        gt_mask (_type_):  [valid_num, H, W]
        cfg (_type_): configuration info
    Returns:
        _type_: _description_
    """
    loss_dict = {}
    if cfg.ce_weight > 0:
        ce_loss = CE_Loss(pred_mask, gt_mask)
        loss_dict['ce_loss'] = (cfg.ce_weight * ce_loss).to(device)
        
    if cfg.focal_weight > 0:
        focal_loss = Focal_Loss(pred_mask, gt_mask)
        loss_dict['focal_loss'] = (cfg.focal_weight * focal_loss).to(device)
    if cfg.dice_weight > 0:
        dice_loss = weighted_Dice_Loss(pred_mask, gt_mask)
        loss_dict['dice_loss'] = (cfg.dice_weight * dice_loss).to(device)
    if cfg.iou_weight > 0:
        iou_loss = IoU_Loss(pred_mask, gt_mask)
        loss_dict['iou'] = (cfg.iou_weight * iou_loss).to(device)
  
    return loss_dict
 