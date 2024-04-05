from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, load_model

from segment_anything import SamPredictor, sam_model_registry

from PIL import Image, ImageDraw, ImageFont
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import cv2
import torch
import os
import numpy as np

from utils import write_masks_to_folder
import yaml
import argparse

def show_mask(masks, image, random_color=True):
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    
    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        
        h, w = mask.shape[-2:]
        mask_image = mask.cpu().reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)
    
    return np.array(annotated_frame_pil)

def detect(args):
    
    print("Loading GroundingDINO Model...")
    groundingdino_model = load_model(args.grounding_config_path, args.grounding_checkpoint_path)
    
    TEXT_PROMPT = args.TEXT_PROMPT
    BOX_TRESHOLD = args.BOX_TRESHOLD
    TEXT_TRESHOLD = args.TEXT_TRESHOLD
    
    annot_image_path = os.path.join(args.images_dir, args.annot_image)
    image_source, image = load_image(annot_image_path)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # Save the image with bounding box
    output_annotated_bbox_path = os.path.join(args.annotation_dir, args.class_name, f"{args.annot_image.split('.')[0]}_bbox.jpg")
    os.makedirs(os.path.join(args.annotation_dir, args.class_name), exist_ok=True)
    cv2.imwrite(output_annotated_bbox_path, annotated_frame)
    
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    
    return boxes_xyxy, annotated_frame

def segment(args, boxes_xyxy, annotated_frame):
    
    mask = None
    
    print("Loading SAM Model...")
    sam = sam_model_registry[args.seg_model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    predictor = SamPredictor(sam)
    
    annot_image_path = os.path.join(args.images_dir, args.annot_image)
    image_source, image = load_image(annot_image_path)
    predictor.set_image(image_source)
    
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to("cuda")
    
    masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )

    output_dir = os.path.join(args.annotation_dir, args.class_name)
    os.makedirs(output_dir, exist_ok=True)
    write_masks_to_folder(masks.cpu().numpy(), os.path.join(output_dir, args.annot_image))
    
    annotated_frame_with_mask = show_mask(masks, annotated_frame)
    result_image_pil = Image.fromarray(annotated_frame_with_mask)
    result_image_pil_rgb = result_image_pil.convert("RGB")
    # Save the image with mask
    result_image_pil_rgb.save(os.path.join(output_dir, f"{args.annot_image.split('.')[0]}_mask.jpg"))
    print("Masks are saved!")
    
    return mask

if __name__ == "__main__":

    # Load YAML configuration
    parser = argparse.ArgumentParser()
    
    with open('config.yaml', 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    for key, value in config_data.items():
        parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} argument from YAML')
    args = parser.parse_args()

    # Get bounding box
    boxes_xyxy, annotated_frame = detect(args)
    # Segment using bounding box
    segment(args, boxes_xyxy, annotated_frame)
