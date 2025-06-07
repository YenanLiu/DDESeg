# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------
# 

import torch
import torch.distributed as dist

from .data_load_aug_avss import AVSSDataLoadImg
from .data_load_aug_vpo import VPODataLoadImg

def build_data_loader_AVSS(cfg, is_train):
    if not is_train:
        dataset = AVSSDataLoadImg(split="test", cfg=cfg, is_train=False)
        if cfg.use_ddp:
            val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.val_per_batch, shuffle=False, 
                                                num_workers=cfg.dataloader.num_workers, sampler=val_sampler, pin_memory=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.val_per_batch, shuffle=True, 
                                                num_workers=cfg.dataloader.num_workers, pin_memory=True, persistent_workers=True)

    else:
        dataset = AVSSDataLoadImg(split="train", cfg=cfg, is_train=True)
        if cfg.use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.train_per_batch, shuffle=False, 
                                                   num_workers=cfg.dataloader.num_workers, sampler=train_sampler, pin_memory=True, persistent_workers=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.train_per_batch, shuffle=False,
                                                num_workers=cfg.dataloader.num_workers, pin_memory=True, persistent_workers=True)
    return dataset, loader

def build_data_loader_VPO(cfg, is_train):
    if not is_train:
        dataset = VPODataLoadImg(split="test", cfg=cfg, is_train=False)
        if cfg.use_ddp:
            val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.val_per_batch, shuffle=False, 
                                                num_workers=cfg.dataloader.num_workers, sampler=val_sampler, pin_memory=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.val_per_batch, shuffle=True, 
                                                num_workers=cfg.dataloader.num_workers, pin_memory=True, persistent_workers=True)

    else:
        dataset = VPODataLoadImg(split="train", cfg=cfg, is_train=True)
        if cfg.use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.train_per_batch, shuffle=False, 
                                                   num_workers=cfg.dataloader.num_workers, sampler=train_sampler, pin_memory=True, persistent_workers=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.train_per_batch, shuffle=False,
                                                num_workers=cfg.dataloader.num_workers, pin_memory=True, persistent_workers=True)
    return dataset, loader