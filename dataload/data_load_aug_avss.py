# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import csv
import os
import torch
import json

from PIL import Image
from dataload.baseloader import BaseDataset
from dataload.data_augmentation import *

def _read_json(json_file):
    if json_file is not None:
        with open(json_file, 'r') as file:
            data = json.load(file)
        return data
    else:
        return None

def read_from_csv(csv_file_path, split, data_dir, task):
    frame_video_paths = []
    
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            r_split = row[5] # train val test
            r_sub_set = row[6] # v1s v1m v2
            if task == "v2":
                if r_split == split:
                    uid = row[1]
                    v_name, img_index = uid.rsplit("_", 1)
                    frame_path = os.path.join(data_dir, r_sub_set, v_name, "frames", img_index + ".jpg")
                    frame_video_paths.append(frame_path)
            else:
                if r_sub_set == task and r_split == split:
                    uid = row[1]
                    v_name, img_index = uid.rsplit("_", 1)
                    frame_path = os.path.join(data_dir, r_sub_set, v_name, "frames", img_index + ".jpg")
                    frame_video_paths.append(frame_path)
                
    return frame_video_paths

class AVSSDataLoadImg(BaseDataset):
    def __init__(self, split, cfg, is_train=False, **kwargs):
        super(AVSSDataLoadImg, self).__init__(cfg.dataloader)
        self.split = split
        self.is_train = is_train

        # fea_save_dir
        self.data_dir = cfg.dataloader.avss_data_dir
        self.img_size = cfg.dataloader.img_size

        self.task = cfg.task
        self.frame_video_paths = read_from_csv(cfg.dataloader.avss_data_file, split, cfg.dataloader.avss_data_dir, cfg.task)
        self.bbox = _read_json(cfg.dataloader.avss_box_json_file)

        if self.is_train:
            self.classical_transform = ClassificalTransformWithMask()
            self.fast_crop_transform = FastRandomCropWithBbox(aspect_ratio_range=(0.75, 1.33), min_area_ratio=0.6)


    def __len__(self):
        return len(self.frame_video_paths)

    def __getitem__(self, index):
        frame_path = self.frame_video_paths[index]
        
        mask_path = frame_path.replace("frames", "labels_semantic").replace(".jpg", ".png")
        aud_path = frame_path.replace("frames", "audios").replace(".jpg", ".wav")

        img = Image.open(frame_path).convert("RGB")
        if img.size[0] == 0 or img.size[1] == 0:
            raise ValueError(f"Image at index {frame_path} has invalid size: {img.size}")
        
        mask = Image.open(mask_path).convert('L')

        if self.bbox is not None:
            img_name = os.path.basename(frame_path).split(".")[0]
            v_name = frame_path.split("/")[-3]
            uid = v_name + "_" + img_name
            bbox = self.bbox[uid]
        else:
            bbox = None
        
        # DATA AUG
        if self.is_train:
            img, mask = self.fast_crop_transform((img, bbox, mask)) # img: ndarray; mask: ndarray
            img, mask = self.classical_transform(img, mask, frame_path) # PIL.Image; PIL.Image

        img_tensor, mask_tensor = process_img_and_mask(img, mask, self.img_size)

        aud_fea = self._load_audio(aud_path)
        aud_tensor = torch.tensor(aud_fea)

        if self.is_train == False:
            return img_tensor, mask_tensor, aud_tensor, frame_path 
        
        return img_tensor, mask_tensor, aud_tensor, frame_path 


