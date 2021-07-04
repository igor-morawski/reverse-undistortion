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
    args = parser.parse_args()
    
    base_config = utils.read_json(op.join(utils.CONFIGS_DIR, args.config))
    for camera in base_config['cameras']:
        print(f"{camera}")
        camera_config = utils.read_json(op.join(utils.CONFIGS_DIR, camera+".json"))
        dataset = Dataset(camera_config)
        dataset.show_stats()
        dataset.filter_by_focal_length(105)
        dataset.show_stats()