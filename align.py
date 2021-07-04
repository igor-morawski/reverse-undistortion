# python align.py --camera=nikon --focal_length=21

import argparse
import glob
import tqdm
import os 
import os.path as op
import PIL.Image
import PIL.ExifTags
import json
import utils
from dataset import Dataset

def 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--camera', type=str)
    parser.add_argument('--focal_length', type=int)
    args = parser.parse_args()
    print(f"{args.camera}")
    camera_config = utils.read_json(op.join(utils.CONFIGS_DIR, args.camera+".json"))
    dataset = Dataset(camera_config)
    print("Original stats:")
    dataset.show_stats()
    dataset.filter_by_focal_length(args.focal_length)
    print(f"Stats for focal length {args.focal_length}:")
    dataset.show_stats()
    