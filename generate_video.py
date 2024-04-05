from tqdm import tqdm
import os
import cv2
import torch
import numpy as np
from PIL import Image

import yaml 
import argparse

from utils import convert_to_image_with_mask

def main(args):
    
    mask_frame_dir = os.path.join(args.video_frames_dir, args.class_name)
    image_dir = args.images_dir
    mask_dir = os.path.join(args.masks_dir, args.class_name)
    video_dir = os.path.join(args.video_dir, args.class_name)
    os.makedirs(mask_frame_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    image_list = sorted(os.listdir(image_dir))
    mask_list = sorted(os.listdir(mask_dir))
    
    first_frame = mask_list[0]
    last_frame = mask_list[-1]
    
    mask_sufix = first_frame.split('.')[-1]
    image_sufix = image_list[0].split('.')[-1]
    if mask_sufix != image_sufix:
        start = image_list.index(first_frame.replace(mask_sufix, image_sufix))
        end = image_list.index(last_frame.replace(mask_sufix, image_sufix)) + 1
    else:
        start = image_list.index(first_frame)
        end = image_list.index(last_frame) + 1
    image_list = image_list[start:end]


    video_frames = []
    for i in tqdm(range(len(image_list))):

        mask = cv2.imread(os.path.join(mask_dir, mask_list[i])) 
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = cv2.imread(os.path.join(image_dir, image_list[i])) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_frame_path = os.path.join(mask_frame_dir, image_list[i])
        image_with_mask = convert_to_image_with_mask(image, mask, mask_frame_path) 
        video_frames.append(image_with_mask)

    height, width, _ = video_frames[0].shape

    video_name = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(os.path.join(video_dir, video_name), fourcc, 10.0, (width, height)) 

    for frame in video_frames:
        frame = frame.astype(np.uint8)
        video.write(frame)

    video.release()

if __name__ == '__main__':
    
    # Load YAML configuration
    parser = argparse.ArgumentParser()
    
    with open('config.yaml', 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    for key, value in config_data.items():
        parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} argument from YAML')
    args = parser.parse_args()
    
    main(args)
