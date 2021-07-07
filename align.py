# python align.py --camera=nikon --focal_length=21

import argparse
import glob
import tqdm
import os 
import os.path as op
import json
import rawpy
import numpy as np
import cv2
from skimage.exposure import match_histograms, equalize_hist
import skimage.filters 
import datetime 
import time 
import copy
import pickle

import utils
from utils import log_msg
from dataset import Dataset, SUPPORTED_ANNO_EXT
import draw_anno
from aligner import Aligner, Transforms


LOG = True
MAP_DIR = "transforms"
VIS_DIR = "visualized"
STEPS_DIR = "steps"
MIN_MATCH_COUNT  = 10



SUPPORTED_RAW_PATTERNS = [b'RGBG']

def image_diff(img1, img2, postprocessing_fncs=None, postprocessing_args=None):
    diff = img1 - img2
    if not postprocessing_fncs:
        return diff
    for fnc, args in zip(postprocessing_fncs, postprocessing_args):
        diff = fnc(diff, **args)
    return diff


def read_avg_green_raw(filepath):
    _img = rawpy.imread(filepath)
    assert _img.color_desc in SUPPORTED_RAW_PATTERNS
    img = _img.raw_image.copy()
    img = np.expand_dims(img,axis=2)
    black_level = _img.black_level_per_channel[0] # assume all black level is the same
    img = (img - black_level)/2**16
    H = img.shape[0]
    W = img.shape[1]
    packed_img = np.concatenate((img[0:H:2, 0:W:2, :],        # R
                              img[0:H:2, 1:W:2, :],           # GR
                              img[1:H:2, 0:W:2, :],           # GB
                              img[1:H:2, 1:W:2, :]), axis=2)  # B
    greens = (packed_img[:, :, 1]+packed_img[:, :, 2])/2 # RGGB
    return greens

def read_rawpy_rgb(filepath):
    _img = rawpy.imread(filepath)
    bgr = _img.postprocess()
    return bgr[:, :, ::-1]

def read_rawpy_grayscale(filepath):
    _img = rawpy.imread(filepath)
    rgb = _img.postprocess()
    return cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2GRAY)

def read_avg_colors_raw(filepath, gamma=True, rgb_weights=[.2126, .7152, .0722]):
    # https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
    _img = rawpy.imread(filepath)
    assert _img.color_desc in SUPPORTED_RAW_PATTERNS
    img = _img.raw_image.copy()
    img = np.expand_dims(img,axis=2)
    black_level = _img.black_level_per_channel[0] # assume all black level is the same
    img = (img - black_level)/2**16
    H = img.shape[0]
    W = img.shape[1]
    Rw, Gw, Bw = rgb_weights
    Gw/=2
    packed_img = np.concatenate((img[0:H:2, 0:W:2, :],        # R
                              img[0:H:2, 1:W:2, :],           # GR
                              img[1:H:2, 0:W:2, :],           # GB
                              img[1:H:2, 1:W:2, :]), axis=2)  # B
    r, gr, gb, b = packed_img[:, :, 0], packed_img[:, :, 1], packed_img[:, :, 2], packed_img[:, :, 3]
    gamma = 1/2.2
    grayscale = Rw * r**gamma + Gw * gr**gamma + Gw * gb**gamma + Bw * b**gamma 
    return grayscale



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--camera', type=str, required=True)
    parser.add_argument('--focal_length', type=int, required=True)
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--document', action="store_true")
    parser.add_argument('--not_log', action="store_true")
    args = parser.parse_args()
    if args.not_log:
        LOG = False
    flags = [args.visualize, args.document, True]
    dirs = [VIS_DIR, STEPS_DIR, MAP_DIR]
    for flag, dir in zip(flags, dirs):
        if flag and not op.exists(dir):
            os.mkdir(dir)
    if args.visualize:
        fl_dir_name = str(args.focal_length).replace(".","_")
        subdirs = [args.camera, fl_dir_name]
        visualize_dir = op.join(VIS_DIR)
        for sub in subdirs:
            visualize_dir = op.join(visualize_dir, sub)
            if not op.exists(visualize_dir):
                os.mkdir(visualize_dir)
    log_msg(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), log=LOG)
    print(f"{args.camera}")
    camera_config = utils.read_json(op.join(utils.CONFIGS_DIR, args.camera+".json"))
    dataset = Dataset(camera_config)
    print("Original stats:")
    dataset.show_stats()
    dataset.filter_by_focal_length(args.focal_length)
    result_filepath = op.join(MAP_DIR, dataset.generate_transform_file_name())
    print(f"Stats for focal length {args.focal_length}:")
    dataset.show_stats()
    transforms = Transforms()
    detector = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    cached_aligner = Aligner(H=dataset.jpeg_h, W=dataset.jpeg_w, window_size=dataset.window_size, stride=dataset.stride)
    cached_addresses_LUT = cached_aligner.addresses_LUT.copy()
    aligners = []
    for idx, sample in tqdm.tqdm(enumerate(dataset.samples)): #XXX
        log_msg(f"Sample {idx}", log=LOG)
        ''' 4. Prepare images. '''
        ''' 4.1. Rawpy+grayscale. ''' 
        raw_img = read_rawpy_grayscale(sample.raw_filepath)
        jpeg_img = cv2.imread(sample.jpeg_filepath, 0)
        if args.document:
            cv2.imwrite("steps/1_1_jpeg.jpg", jpeg_img)
            cv2.imwrite("steps/1_1_raw.jpg", raw_img)
        ''' 4.2.&3. Equalize and match histograms. '''
        jpeg_img = equalize_hist(jpeg_img)
        raw_img = match_histograms(raw_img, jpeg_img)
        if args.document:
            cv2.imwrite("steps/1_2_jpeg.jpg", jpeg_img*255)
            cv2.imwrite("steps/1_2_raw.jpg", raw_img*255)
        ''' 4.3A. Resize RAW'''
        Hj, Wj = jpeg_img.shape
        Hr, Wr = raw_img.shape
        raw_sx = Wj/Wr
        raw_sy = Hj/Hr
        raw_img = cv2.resize(raw_img, None, fx=raw_sx, fy=raw_sy)
        assert raw_img.shape == jpeg_img.shape
        ''' 1.-1. np.UINT8'''
        jpeg_img *= 255
        raw_img *= 255
        assert jpeg_img.max() <= 255
        assert raw_img.max() <= 255
        jpeg_img = jpeg_img.astype(np.uint8)
        raw_img = raw_img.astype(np.uint8)
        ''' 2. Estimate homography'''
        aligner = Aligner(H=dataset.jpeg_h, W=dataset.jpeg_w, window_size=dataset.window_size, stride=dataset.stride, addresses_LUT=cached_addresses_LUT)
        mapping_idx = 0
        for y in range(0, dataset.jpeg_h - dataset.window_size + dataset.window_size, dataset.stride):
            for x in range(0, dataset.jpeg_w - dataset.window_size + dataset.window_size, dataset.stride):
                jpeg_patch = jpeg_img[y:y + dataset.window_size, x:x + dataset.window_size]
                raw_patch = raw_img[y:y + dataset.window_size, x:x + dataset.window_size]
                img1 = jpeg_patch
                img2 = raw_patch
                kp1, des1 = detector.detectAndCompute(img1,None)
                kp2, des2 = detector.detectAndCompute(img2,None)
                if isinstance(des1, type(None)) or isinstance(des2, type(None)) or len(kp1) < 2 or len(kp2) <2:
                    log_msg(f"[WARNING] {sample.file_name} patch {mapping_idx} ignored, (no descriptors)", log=LOG)
                else: 
                    matches = flann.knnMatch(des1,des2,k=2)
                    good = []
                    for m,n in matches:
                        if m.distance < 0.7*n.distance:
                            good.append(m)
                    if len(good)>MIN_MATCH_COUNT:
                        src_pts = np.float32([ (kp1[m.queryIdx].pt[0] + x, kp1[m.queryIdx].pt[1] + y) for m in good ]).reshape(-1,1,2)
                        dst_pts = np.float32([ (kp2[m.trainIdx].pt[0] + x, kp2[m.trainIdx].pt[1] + y) for m in good ]).reshape(-1,1,2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        aligner.mappings[mapping_idx] = M
                    else:
                        log_msg(f"[WARNING] {sample.file_name} patch {mapping_idx} ignored, (not enough good matches)", log=LOG)
                        M, mask = None, None
                mapping_idx +=1
                aligners.append(aligner)
        if args.document:
            cv2.imwrite("steps/0_patch_jpeg.png",jpeg_patch)
            cv2.imwrite("steps/1_patch_raw.png",raw_patch)
    raw_preprocessing_transforms = Transforms()
    raw_preprocessing_transforms.log(np.diag([raw_sx, raw_sy]), "raw", desc=None)
    aggregated_aligner = Aligner(H=dataset.jpeg_h, W=dataset.jpeg_w, window_size=dataset.window_size, stride=dataset.stride, \
        addresses_LUT=cached_addresses_LUT, raw_preprocessing_transforms=raw_preprocessing_transforms, log=LOG)
    for mapping_idx in aggregated_aligner.mappings.keys():
        mappings = []
        for aligner in aligners:
            M = aligner.mappings[mapping_idx]
            if not isinstance(M,type(None)):
                mappings.append(M)
        avg_mapping = np.mean(mappings, axis=0)
        aggregated_aligner.mappings[mapping_idx] = avg_mapping
    aggregated_aligner.save(result_filepath)
    test_aligner = Aligner(H=dataset.jpeg_h, W=dataset.jpeg_w, \
        window_size=dataset.window_size, stride=dataset.stride, load_from=result_filepath)
    # sanity check
    for idx in aggregated_aligner.mappings.keys():
        a1 = aggregated_aligner.mappings[idx]
        a2 = test_aligner.mappings[idx]
        if isinstance(a1, type(None)):
            a1 = np.array([0])
        if isinstance(a2, type(None)):
            a2 = np.array([0])
        if not np.allclose(a1, a2, equal_nan=True): raise Exception(f"{idx}: {a1}, {a2}.")
    assert np.array_equal(aggregated_aligner.addresses_LUT, test_aligner.addresses_LUT)
    log_msg("Saved file validated.")
    if args.visualize:
        for sample in tqdm.tqdm(dataset.samples):
            rgb_raw = read_rawpy_rgb(sample.raw_filepath)
            result = draw_anno.annotate(rgb_raw, dataset.get_anno()[sample.file_name], transform_fnc=test_aligner.apply, draw_org=True)
            cv2.imwrite(op.join(visualize_dir, sample.file_name.split(".")[0]+".JPG"), result)
    print("Done.")
    