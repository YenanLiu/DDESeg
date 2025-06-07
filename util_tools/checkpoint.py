# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import os
import torch
import glob

from omegaconf import OmegaConf
from mmcv.runner import CheckpointLoader
from .custom_logging import get_logger

def load_eval_checkpoint(config, model):
    logger = get_logger()
    logger.info(f'==============> Successfully Loading the Evaluation model from {config.eval_model_path}....................')
    checkpoint = CheckpointLoader.load_checkpoint(config.eval_model_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    logger.info(msg)

    logger.info(f"save training epoch {checkpoint['epoch']}")
    logger.info(f"fscore {checkpoint['fscore']}")
    logger.info(f"miou {checkpoint['miou']}")

def save_checkpoint(config, epoch, model, optimizer, max_metrics, lr_scheduler, save_miou_flag, suffix=''):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'config': OmegaConf.to_container(config, resolve=True),
        'fscore': max_metrics['max_fscore'],
        'miou': max_metrics['max_miou'],
        'epoch': epoch
    }
    logger = get_logger()

    if save_miou_flag:
        exists_pths = glob.glob(config.output + '/*.pth')
        if len(exists_pths) > 0:
            for p in exists_pths:
                os.remove(p)

        filename = f"ck_epoch_{epoch}_maxmiou{max_metrics['max_miou']:.3f}_fscore{max_metrics['max_fscore']:.3f}.pth"

        save_path = os.path.join(config.output, filename)
    
        logger.info(f'{save_path} saving......')
        torch.save(save_state, save_path)
        logger.info(f'{save_path} saved !!!')

"""
Note: In the official released code, I have deleted the fuction call in train.py
    You can recall this function in train.py if needed. -- LC
"""
def resume_training_from_checkpoint(config, model, optimizer=None, scheduler=None):
    logger = get_logger()
    logger.info(f'==============> Resuming training from {config.resume} ....................')

    checkpoint = CheckpointLoader.load_checkpoint(config.resume, map_location='cpu')

    if config.use_ddp:
        new_dict = {}
        for key, value in checkpoint['model'].items():
            new_dict["module." + key] = value
    else:
        new_dict = checkpoint['model']

    # resume model
    msg = model.load_state_dict(new_dict, strict=True)
    logger.info(f"Model loaded: {msg}")

    if 'config' in checkpoint:
        restored_config = OmegaConf.create(checkpoint['config'])
        if 'resume_weight' in restored_config:
            del restored_config.resume_weight
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'epochs' in restored_config.train:
            restored_config.train.epochs = config.train.epochs

        logger.info("Training config loaded from checkpoint.")
        config.merge_with(restored_config) 

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Optimizer resumed.")

    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Scheduler resumed.")
    
    fscore, miou = checkpoint['fscore'], checkpoint['miou']

    return start_epoch, fscore, miou 