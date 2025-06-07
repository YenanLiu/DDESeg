# ---------------------------------------------------------------
# Copyright (c) 2024. All rights reserved.
#
# Written by Chen Liu
# ---------------------------------------------------------------

import torch
import numpy as np
import torchvision.transforms as T
import random
import torchvision.transforms.functional as F
from PIL import Image

def ensure_pil(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img

def ensure_numpy(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    return img

def process_img_and_mask(img, mask, input_size):
    img, mask = map(ensure_pil, [img, mask])
    resize = T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR)
    img = resize(img)
    mask = T.Resize((input_size, input_size), interpolation=T.InterpolationMode.NEAREST)(mask)

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = torch.from_numpy(img).permute(2, 0, 1).float() if isinstance(img, np.ndarray) else T.ToTensor()(img)
    img = norm(img)

    mask = torch.from_numpy(np.array(mask)).long() if not isinstance(mask, torch.Tensor) else mask.long()

    return img, mask

class ClassificalTransformWithMask:
    def __init__(self):

        self.color_jitter = T.ColorJitter(
            brightness=0.2,    
            contrast=0.2,      
            saturation=0.2,     
            hue=0.1             
        )
        self.hflip = T.RandomHorizontalFlip(p=0.5)
        # self.vflip = T.RandomVerticalFlip(p=0.5)
        # self.rotation = T.RandomRotation(degrees=(-30, 30))

    def __call__(self, img, mask, img_path):
        try:
            img, mask = map(ensure_pil, [img, mask])

            if img.size == (0, 0):
                raise ValueError("Input image is empty")

            if mask.size == (0, 0):
                raise ValueError("Input mask is empty")

            if random.random() > 0.5:
                img, mask = map(F.hflip, [img, mask])
            
            img = self.color_jitter(img)

            # if random.random() > 0.5:
            #     img, mask = map(F.vflip, [img, mask])

            # angle = random.uniform(-30, 30)
            # img = F.rotate(img, angle)
            # mask = F.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)

            return img, mask

        except Exception as e:
            print(f"Error during transformation: {e} , path is {img_path}")
            return img, mask

class FastRandomCropWithBbox:
    def __init__(self, aspect_ratio_range=(0.5, 2), min_area_ratio=0.5):
        self.aspect_ratio_range = aspect_ratio_range
        self.min_area_ratio = min_area_ratio

    def __call__(self, img_and_bbox_and_mask):
        img, bbox, mask = img_and_bbox_and_mask
        img, mask = map(ensure_numpy, [img, mask])
        if bbox is None:
            return img, mask
        img, mask = self.crop_image_and_bbox(img, bbox, mask)
        return img, mask

    def crop_image_and_bbox(self, img, bbox, mask):
        height, width = img.shape[:2]
        x_min, y_min = int(bbox[0] * width), int(bbox[1] * height)
        x_max, y_max = int(bbox[2] * width), int(bbox[3] * height)

        bbox_area = (x_max - x_min) * (y_max - y_min)
        aspect_ratio = random.uniform(*self.aspect_ratio_range)
        crop_area = random.uniform(self.min_area_ratio * bbox_area, width * height)
        crop_width = min(int(np.sqrt(crop_area * aspect_ratio)), width)
        crop_height = min(int(np.sqrt(crop_area / aspect_ratio)), height)

        x0, y0 = random.randint(0, width - crop_width), random.randint(0, height - crop_height)
        x1, y1 = x0 + crop_width, y0 + crop_height

        inter_xmin, inter_ymin = max(x_min, x0), max(y_min, y0)
        inter_xmax, inter_ymax = min(x_max, x1), min(y_max, y1)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        if inter_area >= self.min_area_ratio * bbox_area:
            return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        
        return img, mask


# if __name__ == "__main__":
#     import torch
#     import random
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from PIL import Image
#     import torchvision.transforms as T
#     import torchvision.transforms.functional as F

#     img = Image.open('/project/_liuchen/AVSSData/v2/_0je75Y8CmU_30000_40000/frames/1.jpg') 
#     mask = Image.open('/project/_liuchen/AVSSData/v2/_0je75Y8CmU_30000_40000/labels_semantic/1.png') 
#     bbox = [0.0, 0.0, 0.75625, 0.9986111111111111]
#     input_size = 224

#     dataaug_way = "all_aug"  # 可以是 "crop_aug", "classical_aug", "all_aug"

#     fast_crop_transform = FastRandomCropWithBbox()
#     classical_transform = ClassificalTransformWithMask()
    
#     if dataaug_way == "crop_aug":
#         img, mask = fast_crop_transform((img, bbox, mask))
#     elif dataaug_way == "classical_aug":
#         img, mask = classical_transform(img, mask)
#     elif dataaug_way == "all_aug":
#         img, mask = fast_crop_transform((img, bbox, mask))
#         img, mask = classical_transform(img, mask)

#     img_tensor, mask_tensor = process_img_and_mask(img, mask, input_size)

#     img_np = img_tensor.permute(1, 2, 0).numpy()
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
#     mask_np = mask_tensor.numpy()

#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(img_np)
#     ax[0].set_title('Final Processed Image')
#     ax[0].axis('off')

#     ax[1].imshow(mask_np, cmap='gray')
#     ax[1].set_title('Final Processed Mask')
#     ax[1].axis('off')
#     plt.savefig(f"./{dataaug_way}_test.png")
 