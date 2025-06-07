# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import argparse
import gc
import os
import time
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from mmcv.runner import set_random_seed
from mmcv.utils import collect_env
from omegaconf import OmegaConf 
from util_tools import * 
from train import train_worker

warnings.filterwarnings("ignore")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ddp", "-ddp", action="store_true", help="Use DDP for training") # 
    parser.add_argument("--wandb", "-wandb", action="store_true", help="Use Wandb for recording results")# 
    parser.add_argument("--wandb_name", "-wn", type=str, default="test", help="helping distinguish the training change")
    parser.add_argument("--cfg", type=str, default="/root/code/TPMAI_Extension/DDESeg/configs/ddeseg.yaml", help="path to config file")
    parser.add_argument("--task", "-t", type=str, default="v2", choices=('v1s', 'v1m', 'v2', 'VPO-SS', 'VPO-MS', 'VPO-MSMI'), help="path to config file") # choices=('v1s', 'v1m', 'v2')

    parser.add_argument("--num_classes", "-nc", type=int, help="")
    parser.add_argument("--aud_cls_num", "-acn", type=int, default=5, help="The derived audio semantic number.")

    # VOP setting
    parser.add_argument("--vpo_audio_type", "-vpo_at", type=str, default="mono", choices=('mono', 'stereo'), help="")

    # training
    parser.add_argument("--base_lr", "-blr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224, help="")
    parser.add_argument("--epochs", type=int, default=150, help="")

    parser.add_argument("--train_per_batch", "-train_b", default=10, type=int)
    parser.add_argument("--val_per_batch", "-val_b", default=1, type=int)

    parser.add_argument("--output", type=str, default="/project/_liuchen/DDESeg_Log", help="the log saving directory path")
    parser.add_argument("--vis_save_dir", type=str, default="", help="visualization local saving path")
    parser.add_argument("--vis_wandb", action="store_true", help="whether saving visualization into wandb")
    parser.add_argument("--cst_thres", type=float, default=0.1, help="")
 
    # loss
    parser.add_argument("--ce_weight", '-ce', type=float, default=5, help="")
    parser.add_argument("--focal_weight", '-focal', type=float, default=0, help="")    
    parser.add_argument("--dice_weight", '-dice', type=float, default=5, help="")       
    parser.add_argument("--iou_weight", '-iou', type=float, default=2,  help="")   

    # model
    parser.add_argument("--seg_head", '-sh', default='norm', type=str, help="")   

    # for rebuttal
    args = parser.parse_args()
    return args

def main():
    # This code implementation can only support one node with milti-GPUs training
    args = parser_args()
    cfg = get_config(args)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12366'

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    log_dir = os.path.join(cfg.output, args.wandb_name, '{}'.format(time.strftime('%Y%m%d-%H%M%S')))

    if os.path.exists(log_dir):
        log_dir = os.path.join(cfg.output, args.wandb_name, '{}_{}'.format(time.strftime('%Y%m%d-%H%M%S'), np.random.randint(1, 10)))
    
    cfg.output = log_dir
    os.makedirs(cfg.output, exist_ok=True)

    logger = get_logger(cfg)
    logger.info(f"Log dir is {cfg.output}")   

    world_size = torch.cuda.device_count() if cfg.use_ddp else 1
    path = os.path.join(cfg.output, 'config.json')
    OmegaConf.save(cfg, path)
    logger.info(f'Full config saved to {path}')

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.use_ddp :
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, cfg))
    else:
        train_worker(gpu=0, ngpus_per_node=world_size, cfg=cfg)
 
if __name__ == "__main__":
    main()