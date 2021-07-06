# python align.py --camera=nikon --focal_length=21

import argparse
import glob
import tqdm
import os 
import os.path as op
import json
import utils
from dataset import Dataset, SUPPORTED_ANNO_EXT
import rawpy
import numpy as np
import cv2
from skimage.exposure import match_histograms, equalize_hist
import skimage.filters 
import datetime 
import draw_anno

MIN_MATCH_COUNT  = 10

# XXX scaling
# processed_raw_img = skimage.filters.median(raw_img, np.ones([15,15]))
# processed_raw_img = equalize_hist(processed_raw_img, nbins=2**16) 
# raw_img = _match_cumulative_cdf(processed_raw_img, jpeg_img, raw_img)
# raw_img = processed_raw_img


def log_msg(msg):
    print(msg)
    with open('align.log', 'a') as f:
        f.write(msg+"\n")
    return True

class Transforms:
    def __init__(self):
        self.transforms = []
        self.descriptions = []
        self.jpeg_transforms = []
        self.raw_transforms = []
        self.jpeg_descriptions = []
        self.raw_descriptions = []
    
    def log(self, mtx, img, desc=None):
        trnsfm = mtx.copy()
        assert trnsfm.shape == (2,2) or trnsfm.shape == (3,3) 
        if trnsfm.shape == (2,2):
            trnsfm = np.eye(3)
            trnsfm[0:2,0:2] = mtx
        if not desc:
            desc = ""
        else:
            desc = "_" + desc
        desc = img + desc
        if img == "jpeg":
            self.jpeg_transforms.append(trnsfm)
            self.jpeg_descriptions.append(desc)
        elif img == "raw":
            self.raw_transforms.append(trnsfm)
            self.raw_descriptions.append(desc)
        else:
            raise NotImplementedError
        self._update()
    
    def _update(self):
        self.transforms = self._raw_operator(self.raw_transforms) + \
             self._jpeg_operator(self.jpeg_transforms) 
        self.descriptions = self._raw_operator(self.raw_descriptions) + \
             self._jpeg_operator(self.jpeg_descriptions) 
    
    def compose(self):
        t = np.eye(3)
        for m in self.transforms:
            t = m @ t
        return t 
    def _is_l_of_a(self, l):
        for e in l:
            if not isinstance(e, np.ndarray): return False
        return True

    def _jpeg_operator(self, l):
        return l.copy()[::-1]
        
    def _raw_operator(self, l):
        if self._is_l_of_a(l):
            return [np.linalg.inv(a) for a in l]
        else: 
            return l.copy()

    def print(self):
        for idx, (trnsfm, desc) in enumerate(zip(self.transforms, self.descriptions)):
            print(f"{idx}. {desc}: ")
            print(f"{trnsfm}")

class Transform_Fnc:
    def __init__(self, mtx, floor=True) -> None:
        assert mtx.shape == (3,3)
        self.mtx = mtx
        self.floor = floor
    
    def apply(self, x, y) -> list:
        _result = self.mtx @ np.array([x, y, 1])
        if self.floor:
            _result = _result.astype(int)
        return (_result[0], _result[1])


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
    args = parser.parse_args()
    log_msg(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    print(f"{args.camera}")
    camera_config = utils.read_json(op.join(utils.CONFIGS_DIR, args.camera+".json"))
    dataset = Dataset(camera_config)
    print("Original stats:")
    dataset.show_stats()
    dataset.filter_by_focal_length(args.focal_length)
    print(f"Stats for focal length {args.focal_length}:")
    dataset.show_stats()
    # XXX
    transforms = Transforms()
    detector = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    for sample in tqdm.tqdm(dataset.samples[6::7][-1:]):
        ''' 1. Prepare images. '''
        # raw_img = np.array(read_avg_green_raw(sample.raw_filepath)*2**16, dtype=np.uint16)
        ''' 1.1. Rawpy+grayscale. '''
        raw_img = read_rawpy_grayscale(sample.raw_filepath)
        jpeg_img = cv2.imread(sample.jpeg_filepath, 0)
        cv2.imwrite("steps/1_1_jpeg.jpg", jpeg_img)
        cv2.imwrite("steps/1_1_raw.jpg", raw_img)
        ''' 1.2. Equalize and match histograms. '''
        jpeg_img = equalize_hist(jpeg_img)
        raw_img = match_histograms(raw_img, jpeg_img)
        cv2.imwrite("steps/1_2_jpeg.jpg", jpeg_img*255)
        cv2.imwrite("steps/1_2_raw.jpg", raw_img*255)
        ''' 1.3. Downsample each by 4'''
        jpeg_scale = 1/2
        raw_scale = 1/2
        jpeg_img = cv2.resize(jpeg_img, None, fx=jpeg_scale, fy=jpeg_scale)
        raw_img = cv2.resize(raw_img, None, fx=raw_scale, fy=raw_scale)
        transforms.log(jpeg_scale*np.eye(2), img="jpeg")
        transforms.log(raw_scale*np.eye(2), img="raw")
        ''' 1.-1. np.UINT8'''
        if jpeg_img.max() < 2:
            jpeg_img *= 255
        if raw_img.max() < 2:
            raw_img *= 255
        jpeg_img = jpeg_img.astype(np.uint8)
        raw_img = raw_img.astype(np.uint8)
        ''' 2. Estimate homography'''
        img1 = jpeg_img
        img2 = raw_img
        kp1, des1 = detector.detectAndCompute(img1,None)
        kp2, des2 = detector.detectAndCompute(img2,None)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        else:
            log_msg("[WARNING] {} ignored, (not enough good matches)")
            M, mask = None, None
        transforms.log(M, img="jpeg")
    transforms.print()
    mapping = Transform_Fnc(transforms.compose())
    rgb_raw = read_rawpy_rgb(sample.raw_filepath)
    result = draw_anno.annotate(rgb_raw, dataset.get_anno()[sample.file_name], transform_fnc=mapping.apply, draw_org=True)
    cv2.imwrite("result.jpg", result)
    # print(raw_img.shape)
    # print(jpeg_img.shape)
    cv2.imwrite("raw.jpg", raw_img)
    cv2.imwrite("jpeg.jpg", jpeg_img)
    