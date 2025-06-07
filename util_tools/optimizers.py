# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

from torch import optim as optim

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
    
def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    parameters = set_weight_decay(model, {}, {})

    optimizer = optim.AdamW(
        parameters,
        eps=config.optimizer.eps,
        betas=config.optimizer.betas,
        lr=config.base_lr,
        weight_decay=config.weight_decay
    )

    return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]