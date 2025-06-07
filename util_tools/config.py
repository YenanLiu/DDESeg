# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import os.path as osp
from omegaconf import OmegaConf

def load_config(cfg_file):
    # print(cfg_file)
    cfg = OmegaConf.load(cfg_file)
    if '_BASE_' in cfg:
        if isinstance(cfg._BASE_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._BASE_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._BASE_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg

def get_config(args):
    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)
    
    if hasattr(args, 'use_ddp') and args.use_ddp:
        cfg.use_ddp = args.use_ddp

    if hasattr(args, 'wandb') and args.wandb:
        cfg.wandb = args.wandb

    if hasattr(args, 'wandb_name') and args.wandb_name:
        cfg.wandb_name = args.wandb_name

    if hasattr(args, 'task') and args.task:
        cfg.task = args.task

    if hasattr(args, 'output') and args.output:
        cfg.output = args.output

#######################    Training    ##############################
    if hasattr(args, 'epochs') and args.epochs:
        cfg.train.epochs = args.epochs

    if hasattr(args, 'base_lr') and args.base_lr:
        cfg.train.base_lr = args.base_lr

    if hasattr(args, 'train_per_batch') and args.train_per_batch:
        cfg.dataloader.train_per_batch = args.train_per_batch

    if hasattr(args, 'val_per_batch') and args.val_per_batch:
        cfg.dataloader.val_per_batch = args.val_per_batch
    
    if hasattr(args, 'num_workers') and args.num_workers:
        cfg.dataloader.num_workers = args.num_workers

    if hasattr(args, 'print_freq') and args.print_freq:
        cfg.print_freq = args.print_freq

    if hasattr(args, 'num_classes') and args.num_classes:
        cfg.num_classes = args.num_classes 

    if hasattr(args, 'img_size') and args.img_size:
        cfg.dataloader.img_size = args.img_size

    if hasattr(args, 'aud_cls_num') and args.aud_cls_num:
        cfg.align_model.aud_cls_num = args.aud_cls_num

    ############### loss setting ###################
    if hasattr(args, 'ce_weight') and args.ce_weight:
        cfg.loss.ce_weight = args.ce_weight

    if hasattr(args, 'focal_weight') and args.focal_weight:
        cfg.loss.focal_weight = args.focal_weight

    if hasattr(args, 'dice_weight') and args.dice_weight:
        cfg.loss.dice_weight = args.dice_weight

    if hasattr(args, 'iou_weight') and args.iou_weight:
        cfg.loss.iou_weight = args.iou_weight

    ############################  About Evaluation  ################################

    if hasattr(args, 'vis_save_dir') and args.vis_save_dir:
        cfg.evaluate.vis_save_dir = args.vis_save_dir 

    if hasattr(args, 'vis_wandb') and args.vis_wandb:
        cfg.evaluate.vis_wandb = args.vis_wandb 

    ###########################      MODEL Seg Head      ####################################   
    if hasattr(args, 'seg_head') and args.seg_head:
        cfg.seg_head.type = args.seg_head         

    ###########################      VPO dataset        #################################
    if hasattr(args, 'vpo_audio_type') and args.vpo_audio_type:
        cfg.dataloader.vpo_audio_type = args.vpo_audio_type 

    return cfg

 

