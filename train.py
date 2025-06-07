# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import os
import datetime
import time
import warnings
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict
from timm.utils import AverageMeter

from dataload.builder import build_data_loader_VPO, build_data_loader_AVSS
from models import DDESeg
from util_tools import *
from tools import *

warnings.filterwarnings("ignore")

def train_worker(gpu, ngpus_per_node, cfg):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    if cfg.use_ddp:
        rank = int(os.environ.get("RANK", gpu))
        world_size = int(os.environ.get("WORLD_SIZE", ngpus_per_node))
        dist.init_process_group(backend='nccl', init_method='env://', world_size=ngpus_per_node, rank=rank)
        dist.barrier()
        print(f"Process group initialized. Rank: {rank}, World Size: {ngpus_per_node}")
    else:
        rank = 0

    logger = get_logger()
    if cfg.wandb:
        initialize_wandb(cfg, rank)

    if rank == 0:
        logger.info(f"Available devices: {torch.cuda.device_count()}")
        logger.info(f"Training started on device: {device}")

    # init the dataloader
    if "VPO" in cfg.task:
        train_dataset, train_loader = build_data_loader_VPO(cfg, is_train=True)
        val_dataset, val_loader = build_data_loader_VPO(cfg, is_train=False)

    if "v1s" in cfg.task or "v1m" in cfg.task or "v2" in cfg.task:
        train_dataset, train_loader = build_data_loader_AVSS(cfg, is_train=True)
        val_dataset, val_loader = build_data_loader_AVSS(cfg, is_train=False)

    model = DDESeg(
        num_classes=cfg.num_classes,
        embedding_dim=cfg.seg_head.embedding_dim,
        audio_cfg=cfg.audio_model,
        visual_cfg=cfg.vis_model,
        av_cfg=cfg.align_model,
        visual_pretrain=cfg.vis_pretrain,
        seg_head_cfg=cfg.seg_head,
        task=cfg.task
    ).to(device) 

    if rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("==> Total params: %.2fM" % (n_parameters / 1e6))

    if cfg.use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    # optimizer & learning scheduler
    optimizer = build_optimizer(cfg.train, model) 
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(train_loader))

    max_fscore = max_miou = 0.0    
    max_metrics = {'max_fscore': max_fscore, 'max_miou': max_miou}

    start_ttime = time.time()
    for epoch in range(0, cfg.train.epochs):
        if cfg.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"****************     Starting training the {epoch} epoch       *********************")
            logger.info(f"The amount of training data: {len(train_dataset)}, validation data: {len(val_dataset)}")
        
        # Training 
        start_time = time.time()
        train_one_epoch(cfg, model, train_loader, optimizer, lr_scheduler, epoch, device)
        end_time = time.time()
        if rank == 0:
            print("Training one epoch time:", end_time - start_time)
        torch.cuda.empty_cache() 

        if cfg.use_ddp:
            dist.barrier()

        # Evaluation
        metric_dict = evaluation(cfg, model, val_loader, device, epoch=epoch)
        torch.cuda.empty_cache()
        
        if rank == 0:           
            miou_save_flag = metric_dict["miou"] > max_metrics["max_miou"]
            if miou_save_flag :
                max_metrics["max_fscore"] = metric_dict["f_score"]
                max_metrics["max_miou"] = metric_dict["miou"]
                logger.info(f"Saving new best model at epoch {epoch}: mIoU = {metric_dict['miou']:.4f}")
                save_checkpoint(cfg, epoch, 
                    model.module if cfg.use_ddp else model,
                    optimizer, 
                    max_metrics, 
                    lr_scheduler, 
                    miou_save_flag
                )   

        if cfg.use_ddp:
            dist.barrier()

    total_time = time.time() - start_ttime
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        logger.info('Training and Eval time {}'.format(total_time_str))
    
def train_one_epoch(config, model, data_loader, optimizer, lr_scheduler, epoch, device):
    rank = dist.get_rank() if config.use_ddp else 0
    logger = get_logger(config) if rank == 0 else logging.getLogger("rank_dummy")

    if rank == 0:
        logger.info(f'*******************   Training at epoch {epoch}   ***********************')

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    
    model.train()
    for idx, batch_data in enumerate(data_loader):

        frame_tensors, gt_masks, aud_tensors, _ = batch_data
        frame_tensors = frame_tensors.to(device, non_blocking=True)
        aud_tensors = aud_tensors.to(device, non_blocking=True)
        gt_masks = gt_masks.to(device, non_blocking=True)

        if config.num_classes == 1:
            gt_masks = (gt_masks > 0).long()

        pred_masks = model(img_input=frame_tensors, aud_input=aud_tensors)

        loss_dict = loss_calculation(pred_masks, gt_masks, config.loss, device)
        if rank == 0:
            for loss_name, value in loss_dict.items():
                log_vars_meters[f"{loss_name}"].update(value.item())

        if config.use_ddp:
            for key, value in loss_dict.items():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                loss_dict[key] = value / dist.get_world_size()

        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        batch_time.update(time.time() - end)
        end = time.time() 
        loss_meter.update(total_loss.item())

        if rank == 0:
            lr = optimizer.param_groups[0]['lr']
            log_vars_str = ' '.join(f'{n} {m.val:.4f} ({m.avg:.4f})' for n, m in log_vars_meters.items())
            logger.info(f'Train: [{epoch}/{int(config.train.epochs)}][{idx}/{num_steps}] '
                        f'total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'{log_vars_str} '
                        f'lr {lr:.6f}')

            if idx % 20 == 0:
                allocated = torch.cuda.memory_allocated(device=device) / 1024**2
                reserved = torch.cuda.memory_reserved(device=device) / 1024**2
                logger.info(f"[Epoch {epoch} Batch {idx}] GPU Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

            if config.wandb and idx % 20 == 0:
                log_stat = {f'iter/train_{n}': m.avg for n, m in log_vars_meters.items()}
                log_stat['iter/train_total_loss'] = loss_meter.avg
                log_stat['iter/learning_rate'] = lr
                global_step = epoch * len(data_loader) + idx
                wandb_log(log_stat, step=global_step)

    epoch_time = time.time() - start
    if rank == 0:
        logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
        logger.info(f"Avg Training Loss on epoch {epoch} is: {loss_meter.avg:.5f}")        

def compute_metrics_ddp(use_ddp, total_inter, total_union, total_tp, total_fp, total_fn, epsilon=1e-10):
    # all_reduce sums
    if use_ddp:
        total_inter = safe_all_reduce(total_inter, fallback=torch.zeros_like(total_inter))
        total_union = safe_all_reduce(total_union, fallback=torch.ones_like(total_union))
        total_tp = safe_all_reduce(total_tp, fallback=torch.zeros_like(total_tp))
        total_fp = safe_all_reduce(total_fp, fallback=torch.zeros_like(total_fp))
        total_fn = safe_all_reduce(total_fn, fallback=torch.zeros_like(total_fn))
    
    # avoid division by zero
    valid_class_mask = total_union > 0
    
    # mIoU
    iou_per_class = torch.zeros_like(total_inter)
    iou_per_class[valid_class_mask] = total_inter[valid_class_mask] / (total_union[valid_class_mask] + epsilon)

    if valid_class_mask.sum() > 0:
        mean_iou = torch.mean(iou_per_class[valid_class_mask])
    else:
        mean_iou = torch.tensor(0.0, device=total_inter.device)
    
    # Precision, Recall, F-score
    precision = torch.zeros_like(total_tp)
    recall = torch.zeros_like(total_tp)
    fscore = torch.zeros_like(total_tp)
    fscore_03 = torch.zeros_like(total_tp)
    
    precision[valid_class_mask] = total_tp[valid_class_mask] / (total_tp[valid_class_mask] + total_fp[valid_class_mask] + epsilon)
    recall[valid_class_mask] = total_tp[valid_class_mask] / (total_tp[valid_class_mask] + total_fn[valid_class_mask] + epsilon)
    fscore[valid_class_mask] = 2 * (precision[valid_class_mask] * recall[valid_class_mask]) / (precision[valid_class_mask] + recall[valid_class_mask] + epsilon)
    fscore_03[valid_class_mask] = (1 + 0.3) * (precision[valid_class_mask] * recall[valid_class_mask]) / (0.3 * precision[valid_class_mask] + recall[valid_class_mask] + epsilon)
    
    mean_fscore = torch.mean(fscore[valid_class_mask])
    mean_fscore_03 = torch.mean(fscore_03[valid_class_mask])
    
    return {
        "mean_iou": mean_iou.item(),
        "mean_fscore": mean_fscore.item(),
        "mean_fscore_03": mean_fscore_03.item(),
        "per_class_iou": iou_per_class.cpu(),
        "per_class_fscore": fscore.cpu(),
        "per_class_fscore_03": fscore_03.cpu()
    }
 
@torch.no_grad()
def evaluation(config, model, data_loader, device, epoch):
    vis_dict = {}
    rank = dist.get_rank() if config.use_ddp else 0
    logger = get_logger(config) if rank == 0 else logging.getLogger("rank_dummy")
     
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)
    
    # for semantic evaluation
    sem_bin_num = config.num_classes if config.num_classes > 1 else config.num_classes + 1 
    total_sem_inter, total_sem_union = torch.zeros(sem_bin_num, device=device), torch.zeros(sem_bin_num, device=device)
    total_sem_tp, total_sem_fp, total_sem_fn = torch.zeros(sem_bin_num, device=device), torch.zeros(sem_bin_num, device=device), torch.zeros(sem_bin_num, device=device)

    vis_dict = {}

    end = time.time()
    for idx, batch_data in enumerate(data_loader):
        frame_tensors, gt_masks, aud_tensors, frame_paths = batch_data
        frame_tensors = frame_tensors.to(device, non_blocking=True)
        aud_tensors = aud_tensors.to(device, non_blocking=True)
        gt_masks = gt_masks.to(device, non_blocking=True)
            
        if config.num_classes == 1:
            gt_masks = (gt_masks > 0).long()
                
        pred_masks = model(img_input=frame_tensors, aud_input=aud_tensors)
    
        loss_dict = loss_calculation(pred_masks, gt_masks, config.loss, device)
        
        if rank == 0:
            for loss_name, value in loss_dict.items():
                log_vars_meters[f"s_{loss_name}"].update(value.item())

        loss = sum(loss_dict.values())

        if config.use_ddp:
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        loss_meter.update(reduced_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if config.num_classes == 1:
            pred_masks = torch.sigmoid(pred_masks).squeeze(1) # [valid_num, C, H, W]
            pred_masks = (pred_masks > 0.5).float()  # [valid_num, H, W]
        else:
            pred_masks = torch.softmax(pred_masks, dim=1) # [valid_num, C, H, W]
            pred_masks = torch.argmax(pred_masks, dim=1) # [valid_num, H, W]

        sem_inter, sem_union = compute_iou_per_class(pred_masks, gt_masks, config.num_classes)
        sem_tp, sem_fp, sem_fn = compute_fscore_per_class(pred_masks, gt_masks, config.num_classes)
        total_sem_tp += sem_tp
        total_sem_fp += sem_fp
        total_sem_fn += sem_fn
        total_sem_inter += sem_inter
        total_sem_union += sem_union

        if rank == 0:
            global_step = epoch * len(data_loader) + idx
            log_vars_str = ' '.join(f'{n} {m.val:.4f} ({m.avg:.4f})' for n, m in log_vars_meters.items())
            logger.info(f'Test: [{idx}/{len(data_loader)}] Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) {log_vars_str}')

            if config.wandb and idx % 20 == 0:
                wandb_iter_log = {f'iter/val_{n}': m.avg for n, m in log_vars_meters.items()}
                wandb_iter_log['iter/val_total_loss'] = loss_meter.avg
                wandb_log(wandb_iter_log, step=global_step)

        # Visualization
        if rank == 0 and (len(config.evaluate.vis_save_dir) > 0 or config.evaluate.vis_wandb):
            valid_paths = frame_paths
            vis_dict = vis_masks(pred_masks, gt_masks, valid_paths, task=config.task, vis_dict=vis_dict, vis_save_dir=config.vis_save_dir if len(config.evaluate.vis_save_dir) > 0 else None)
            if config.vis_wandb:
                wandb_log_image_dict(vis_dict, step=epoch)

    sem_metrics = compute_metrics_ddp(config.use_ddp, total_sem_inter, total_sem_union, total_sem_tp, total_sem_fp, total_sem_fn, epsilon=1e-10)

    if rank == 0:
        logger.info(f"Avg Validation Loss on epoch {epoch}: {loss_meter.avg:.5f}")
        logger.info(f"Semantic metrics: mIoU {sem_metrics['mean_iou']:.4f}, F-score {sem_metrics['mean_fscore']:.4f}, F0.3-score {sem_metrics['mean_fscore_03']:.4f}")

        if config.wandb:
            wandb_epoch_log = {
                "epoch": epoch,
                "epoch/val_loss": loss_meter.avg,
            }
            wandb_epoch_log.update({
                "epoch/mIoU": sem_metrics["mean_iou"],
                "epoch/fscore": sem_metrics["mean_fscore"],
                "epoch/fscore_03": sem_metrics["mean_fscore_03"],
            })
            wandb_log(wandb_epoch_log, step=epoch)

    return {"miou": sem_metrics["mean_iou"], "f_score": sem_metrics["mean_fscore_03"]}

