# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .builder import *
from .seg_head import *
from .aud_enhancer import TokenFeatureEnhancer

class DDESeg(nn.Module):
    def __init__(self, *, num_classes=71,
                embedding_dim=256,
                audio_cfg=None,
                visual_cfg=None,
                av_cfg=None,
                visual_pretrain=None,
                seg_head_cfg=None,
                task=None,):

        super(DDESeg, self).__init__()

        ## audio backbone
        self.audio_model = build_audio_model(audio_cfg)
        if self.training:
            self.audio_model.load_state_dict(torch.load("/data/AVSSData/HTSAT_AudioSet_Saved_1_clean.ckpt"), strict=False)
            print("Loading audio pretrain model Successful!!!")
        
        self.__set_all_params_frozen(self.audio_model)

        # audio token enhancer
        self.aud_enhancer = TokenFeatureEnhancer(embed_dim=768, k=av_cfg.aud_cls_num)

        ## visual backbone
        if visual_cfg.type == "FAVS":
            self.backbone = build_favs_model(visual_cfg)       
            in_channels = visual_cfg.in_channels

        if visual_cfg.type == "SwinEnhance":
            self.backbone = build_swin_enhancer_model(visual_cfg)
            in_channels = self.backbone.num_features # [96, 192, 384, 768]

        if visual_pretrain is not None and self.training:
            vis_state_dict = load_visual_pretrain(visual_pretrain)
            self.backbone.load_state_dict(vis_state_dict, strict=False)
            print("Loading visual pretrain model Successful!!!")

        self.stages = len(in_channels)

        # the segmentation head
        if seg_head_cfg.type == "norm":
            self.decode_head = SegHead(num_classes, in_channels, embedding_dim)
        elif seg_head_cfg.type == "weighted":
            self.decode_head = WeightedSegHead(num_classes, in_channels, embedding_dim)
        else:
            raise ValueError(f"Cannot construct the type of {seg_head_cfg.type} head!!!!")
        
        # memory bank
        if "VPO" in task:
            aud_bank_path = av_cfg.vpo_aud_bank
        elif "v1s" in task or "v1m" in task or "v2" in task:
            aud_bank_path = av_cfg.avss_aud_bank
        else:
            raise ValueError(f"Cannot find the audio bank path for task {task}!!!!")

        self.aud_bank = torch.tensor(np.load(aud_bank_path))

    def __set_all_params_frozen(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def _set_trainable_audio_params(self, audio_model):     
        print("Setting Frozen or Trainable Params for Audio Model!!!!!!")   
        for _, param in audio_model.named_parameters():
            param.requires_grad = True

    def forward_features(self, img_input, aud_input):
        B = img_input.shape[0]
        aud_bank = self.aud_bank.to(aud_input.device)
        aud_fea = self.audio_model(aud_input)["latent_output"]  

        _, aud_enhanced_token = self.aud_enhancer(aud_fea, aud_bank) # [B, K, 768]
        fuse_fea_dict = self.backbone(img_input, aud_enhanced_token)

        outs = []

        for s in range(self.stages):
            fuse_fea, H, W = fuse_fea_dict[f'fea_{s}'], fuse_fea_dict[f'H_{s}'], fuse_fea_dict[f'W_{s}']
            fuse_fea = fuse_fea.view(B, H, W, -1)
            fuse_fea = fuse_fea.permute(0, 3, 1, 2).contiguous()

            outs.append(fuse_fea)

        return outs 

    def forward(self, img_input, aud_input):
        H, W = img_input.size(2), img_input.size(3)
        x = self.forward_features(img_input, aud_input)

        mask = self.decode_head.forward(x)
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=True)

        return mask 
    
def load_visual_pretrain(pre_train_path):
    def __state_dict_clean(ori_state_dict):
        
        state_dict = {key.replace("backbone.", ""): torch.tensor(value) if isinstance(value, np.ndarray) else value for key, value in ori_state_dict.items()}
        return state_dict
    
    if "ade" in pre_train_path:
        ori_state_dict = torch.load(pre_train_path)['state_dict']
        state_dict = __state_dict_clean(ori_state_dict)

        if 'decode_head.linear_pred.weight' in state_dict:
            del state_dict['decode_head.linear_pred.weight']
    
        if 'decode_head.linear_pred.bias' in state_dict:
            del state_dict['decode_head.linear_pred.bias']

    else:
        ori_state_dict = torch.load(pre_train_path)
        state_dict = __state_dict_clean(ori_state_dict)
    return state_dict

 