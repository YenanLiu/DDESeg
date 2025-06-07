# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

def build_scheduler(config, optimizer, num_steps, start_epoch=0):
    """
    Args:
        num_steps: step amount of each epoch
        start_epoch: resuming epoch
    Returns:
        scheduler
    """
    warmup_steps = int(config.warmup_epochs * num_steps)
    start_step = start_epoch * num_steps

    def warmup_lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs * num_steps,
            eta_min=config.min_lr,
            last_epoch=start_step - warmup_steps - 1 if start_step > warmup_steps else -1
    )

    if config.warmup_epochs > 0:
        warmup_scheduler = LambdaLR(
            optimizer, 
            lr_lambda=warmup_lr_lambda,
            last_epoch=start_step - 1 if start_step <= warmup_steps else warmup_steps - 1
        )
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, base_scheduler], 
            milestones=[warmup_steps],
            last_epoch=start_step - 1
        )
        return scheduler
    else:
        return base_scheduler