# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import os
import torch
import json
import pandas as pd

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

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} is not exists!!!")

def shuffle_lists(frame_paths, mask_paths, audio_paths):
    combined = list(zip(frame_paths, mask_paths, audio_paths))
    random.shuffle(combined)

    frame_paths_shuffled, mask_paths_shuffled, audio_paths_shuffled = zip(*combined)

    return list(frame_paths_shuffled), list(mask_paths_shuffled), list(audio_paths_shuffled)

def read_from_csv(csv_file_path, split, data_dir, task_type, audio_type="stereo"): # mono
    frame_paths, mask_paths, audio_paths = [], [], []
    
    df = pd.read_csv(csv_file_path)
    filtered_df = df[(df['split'] == split) & (df['type'] == task_type)]

    for index, row in filtered_df.iterrows():
        uid = row[0]
        frame_basename = row[1]
        mask_basename = row[2]

        frame_path = os.path.join(data_dir, uid, "frames", frame_basename)
        mask_path = os.path.join(data_dir, uid, "labels_semantic", mask_basename)

        if audio_type == "stereo":
            audio_path = os.path.join(data_dir, uid, "stereo_audio.wav")
        else:
            audio_path = os.path.join(data_dir, uid, "mono_audio.wav")

        check_file_exists(frame_path)
        check_file_exists(mask_path)
        check_file_exists(audio_path)

        frame_paths.append(frame_path)
        mask_paths.append(mask_path)
        audio_paths.append(audio_path)
                
    return frame_paths, mask_paths, audio_paths

class VPODataLoadImg(BaseDataset):
    def __init__(self, split, cfg, is_train=False, **kwargs):
        super(VPODataLoadImg, self).__init__(cfg.dataloader)
        self.split = split
        self.is_train = is_train

        # fea_save_dir
        self.data_dir = cfg.dataloader.vpo_data_dir
        self.img_size = cfg.dataloader.img_size

        self.task = cfg.task
        frame_paths, mask_paths, audio_paths = read_from_csv(cfg.dataloader.vpo_datafile, split, self.data_dir, cfg.task, cfg.dataloader.vpo_audio_type)

        self.classical_transform = ClassificalTransformWithMask()
        self.fast_crop_transform = FastRandomCropWithBbox(aspect_ratio_range=(0.75, 1.33), min_area_ratio=0.6)
        self.frame_paths, self.mask_paths, self.audio_paths = shuffle_lists(frame_paths, mask_paths, audio_paths)
        self.bbox = _read_json(cfg.dataloader.vpo_box_json_file)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, index):
        frame_path = self.frame_paths[index]
        mask_path = self.mask_paths[index]
        audio_path = self.audio_paths[index]
        if not os.path.exists(frame_path):
            print(f"img path is not exists {frame_path}")

        if not os.path.exists(mask_path):
            print(f"mask path is not exists {mask_path}")
        
        img = Image.open(frame_path).convert("RGB") 
        mask = Image.open(mask_path).convert('L')

        if self.bbox is not None:
            img_name = os.path.basename(frame_path).split(".")[0]
            v_name = frame_path.split("/")[-3]
            uid = v_name + "_" + img_name
            bbox = self.bbox[mask_path]
        else:
            bbox = None
        
        # DATA AUG
        if self.is_train:
            img, mask = self.fast_crop_transform((img, bbox, mask)) # img: ndarray; mask: ndarray
            img, mask = self.classical_transform(img, mask, frame_path) # PIL.Image; PIL.Image
 
        img_tensor, mask_tensor = process_img_and_mask(img, mask, self.img_size)

        aud_fea = self._load_audio(audio_path)
        aud_tensor = torch.tensor(aud_fea)

        if self.is_train == False:
            return img_tensor, mask_tensor, aud_tensor, frame_path 
        return img_tensor, mask_tensor, aud_tensor 
 
