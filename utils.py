import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
from torchvision.utils import draw_segmentation_masks
import torch
    
def write_masks_to_folder(masks, mask_path) -> None:
    
    for i, (mask_data) in enumerate(masks):
        dimensions = mask_data.shape
        if len(dimensions) == 2:
            mask = mask_data[:,:]
            cv2.imwrite(mask_path, mask * 255)
            return 
        else:
            mask = mask_data[0, :,:]
            if i == 0:
                integrated_mask = mask
            else:
                integrated_mask = np.logical_or(integrated_mask, mask).astype(np.uint8)
    cv2.imwrite(mask_path, integrated_mask * 255)
        
def convert_to_image_with_mask(image, mask, mask_frame_path, score = None):

    if mask.ndim == 2:
        mask_expanded = np.repeat(mask[..., np.newaxis], 3, axis=2)
    else:
        mask_expanded = mask
    
    image_with_mask = draw_segmentation_masks(torch.from_numpy(image).permute(2,0,1), 
                                                masks=torch.tensor(mask_expanded, dtype = bool).permute(2,0,1), 
                                                alpha=0.15)
    image_with_mask = image_with_mask.permute(1,2,0)
    image_with_mask = image_with_mask.numpy()
    image_with_mask = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)

    cv2.imwrite(mask_frame_path, image_with_mask)

    return image_with_mask