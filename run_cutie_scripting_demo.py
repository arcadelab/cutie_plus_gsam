import os

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

import yaml
import argparse


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main(args):
    # obtain the Cutie model with default parameters -- skipping hydra configuration
    cutie = get_default_model()
    # Typically, use one InferenceCore per video
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    image_path = args.images_dir
    # ordering is important
    images = sorted(os.listdir(image_path))

    # mask for the first frame
    # NOTE: this should be a grayscale mask or a indexed (with/without palette) mask,
    # and definitely NOT a colored RGB image
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html: mode "L" or "P"
    mask = Image.open(os.path.join(args.annotation_dir, args.class_name, args.annot_image))
    assert mask.mode in ['L', 'P']

    # palette is for visualization
    palette = mask.getpalette()

    # the number of objects is determined by counting the unique values in the mask
    # common mistake: if the mask is resized w/ interpolation, there might be new unique values
    objects = np.unique(np.array(mask))
    # background "0" does not count as an object
    objects = objects[objects != 0].tolist()

    mask = torch.from_numpy(np.array(mask)).cuda()
    mask_dir = os.path.join(args.masks_dir, args.class_name)
    os.makedirs(mask_dir, exist_ok=True)

    for ti, image_name in enumerate(images):
        # load the image as RGB; normalization is done within the model
        image = Image.open(os.path.join(image_path, image_name))
        image = to_tensor(image).cuda().float()

        if ti == 0:
            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            output_prob = processor.step(image, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory
            output_prob = processor.step(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob)

        # visualize prediction
        mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        # mask.putpalette(palette)
        save_path = os.path.join(mask_dir, image_name)
        mask.save(save_path)
        # mask.show()  # or use mask.save(...) to save it somewhere


if __name__ == "__main__":
    # Load YAML configuration
    parser = argparse.ArgumentParser()
    
    with open('config.yaml', 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    for key, value in config_data.items():
        parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} argument from YAML')
    args = parser.parse_args()
    
    main(args)
