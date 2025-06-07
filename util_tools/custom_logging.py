# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import logging
import torch
import torch.distributed as dist
import logging
import os
import sys
from omegaconf import OmegaConf

logger_name = None
_global_wandb = None

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def initialize_wandb(cfg, rank=0):
    global _global_wandb

    if rank != 0:
        print(f"Rank {rank}: Skipping wandb init.")
        return

    required_fields = ['wandb', 'project', 'wandb_name', 'output']
    for field in required_fields:
        if not hasattr(cfg, field):
            raise ValueError(f"Missing required config field: '{field}'")

    if _global_wandb is None and cfg.wandb:
        import wandb   
        _global_wandb = wandb
        _global_wandb.init(
            project=cfg.project,
            name=cfg.wandb_name,
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        print("Wandb initialized successfully")
    elif _global_wandb is None:
        print("Wandb initialization skipped as cfg.wandb is False")
    else:
        print("Wandb was already initialized")


def get_wandb():
    global _global_wandb
    return _global_wandb

def wandb_log(data_dict, step=None):
    wb = get_wandb()
    if wb is not None and hasattr(wb, 'log') and is_main_process():
        try:
            wb.log(data_dict, step=step)
        except Exception as e:
            print(f"[Wandb Log] Failed to log data: {e}")

def wandb_log_image_dict(image_dict, step=None):
    wb = get_wandb()
    if wb is not None and hasattr(wb, 'log') and is_main_process():
        try:
            wb.log(image_dict, step=step)
        except Exception as e:
            print(f"[Wandb Log] Failed to log images at step {step}: {e}")

def get_logger(cfg=None, log_level=logging.INFO, rank=0):
    if rank != 0:
        logging.getLogger().setLevel(logging.WARN)
        return logging.getLogger("silent")

    name = getattr(cfg, 'wandb_name', 'experiment') if cfg else 'experiment'
    output = getattr(cfg, 'output', './logs') if cfg else './logs'
    os.makedirs(output, exist_ok=True)
    log_path = os.path.join(output, 'log.txt')

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False 

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_fmt = logging.Formatter(fmt='[%(asctime)s %(name)s] %(levelname)s: %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(log_level)

    # Stream handler (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    console_fmt = logging.Formatter(fmt='[%(asctime)s %(name)s] %(levelname)s: %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(console_fmt)
    stream_handler.setLevel(log_level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def setup_logging(filename, resume=False):
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.handlers.clear()

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=filename, mode='a' if resume else 'w')

    root_logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    
    return total_norm