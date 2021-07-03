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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='base.json')
    parser.add_argument('--filter_anno', action="store_true")
    parser.add_argument('--focal_length', type=int, default=None)
    args = parser.parse_args()
    
    base_config = utils.read_json(args.config)
    for camera in base_config['cameras']:
        print(f"{camera}")
        camera_config = utils.read_json(camera+".json")
        dataset = Dataset(camera_config)
        
    if args.filter_anno:
        # TODO: filter anno
        print("TODO")
        assert args.focal_length 
        pass