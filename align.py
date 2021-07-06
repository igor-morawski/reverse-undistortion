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
import time 

MIN_MATCH_COUNT  = 10

# XXX scaling
# processed_raw_img = skimage.filters.median(raw_img, np.ones([15,15]))
# processed_raw_img = equalize_hist(processed_raw_img, nbins=2**16) 
# raw_img = _match_cumulative_cdf(processed_raw_img, jpeg_img, raw_img)
# raw_img = processed_raw_img

class Aligner:
    def __init__(self, H, W, window_size : int, stride, addresses_LUT = None, mappings = None, raw_preprocessing_transforms = None):
        ''' Maps are numbered in C-order (like np.flatten)'''
        assert isinstance(window_size, int)
        assert stride <= window_size
        assert not H % window_size 
        assert not W % window_size
        assert not H % stride 
        assert not W % stride

        self.raw_preprocessing_transforms = raw_preprocessing_transforms
        if self.raw_preprocessing_transforms:
            #raw_preprocessing_transforms composed
            self._rptc = np.linalg.inv(self.raw_preprocessing_transforms.compose())
        else: 
            self._rptc = np.eye(3)

        self.H = H
        self.W = W
        self.window_size = window_size
        self.stride = stride
    
        self._maps_n = (self._apply_conv_formula(self.H + self.window_size, self.window_size, padding=0, stride=self.stride, assert_divisible=True)) * \
            (self._apply_conv_formula(self.W + self.window_size, self.window_size, padding=0, stride=self.stride, assert_divisible=True))

        if isinstance(addresses_LUT, type(None)):
            print("[WARNING] Constructing LUTs is costly; cache if possible.")
            self.addresses_LUT = self._construct_LUT_adresses()
        else:
            assert addresses_LUT.shape == (self.H, self.W)
            self.addresses_LUT = addresses_LUT
        assert self.addresses_LUT.max() + 1 == self._maps_n #
        if not mappings:
            self.mappings = {i : None for i in range(self._maps_n)}
        assert max(self.mappings.keys()) == self._maps_n - 1
        
    
    def _apply_conv_formula(self, input, kernel, padding, stride, assert_divisible=False):
        result = int(int(input-kernel+2*padding)/stride)
        if assert_divisible:
            result == (input-kernel+2*padding)/stride
        return result

    def loop_over_locations(self, fnc, kwargs):
        for y in range(0, self.H - self.window_size + self.window_size, self.stride):
            for x in range(0, self.W - self.window_size + self.window_size, self.stride):
                fnc(x=x, y=y, **kwargs)

    def _construct_LUT_adresses(self):
        distances = float("inf")*np.ones((self.H, self.W))
        addresses = -1*np.ones((self.H, self.W), dtype=int)
        ys = -1*np.ones((self.H, self.W))
        xs = -1*np.ones((self.H, self.W))
        for i in range(self.H):
            for j in range(self.W):
                ys[i, j] = i
                xs[i, j] = j 
        idx = 0
        for y in range(0, self.H - self.window_size + self.window_size, self.stride):
            for x in range(0, self.W - self.window_size + self.window_size, self.stride):
                distances_window = distances[y:y + self.window_size, x:x + self.window_size]
                if not distances_window.shape == (self.window_size, self.window_size):
                    dws = distances_window.shape
                    ws = self.window_size
                    assert (dws[0] == ws and dws[1] == ws//2) or (dws[1] == ws and dws[0] == ws//2)  or (dws[1] == ws//2 and dws[0] == ws//2)
                dist_y = ys[y:y + self.window_size, x:x + self.window_size] - y + self.window_size//2
                dist_x = xs[y:y + self.window_size, x:x + self.window_size] - x + self.window_size//2
                dist = np.sqrt(dist_y**2 + dist_x**2)
                indices = np.where(dist < distances_window)
                distances[indices[0]+y, indices[1]+x] = dist[indices]
                addresses[indices[0]+y, indices[1]+x] = idx
                idx += 1
        assert (addresses >= 0).all()
        return addresses

        
    def apply(self, x, y, floor=True) -> list:
        mapping_idx = self.addresses_LUT[y, x] 
        mapping = self.mappings[mapping_idx]
        _result = mapping @ np.array([x, y, 1])
        _result = self._rptc @ _result
        if floor:
            _result = _result.astype(int)
        return (_result[0], _result[1])

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
    cached_aligner = Aligner(H=dataset.jpeg_h, W=dataset.jpeg_w, window_size=dataset.window_size, stride=dataset.stride)
    cached_addresses_LUT = cached_aligner.addresses_LUT.copy()
    aligners = []
    for sample in tqdm.tqdm(dataset.samples[3:4]): #XXX
        ''' 4. Prepare images. '''
        ''' 4.1. Rawpy+grayscale. ''' 
        raw_img = read_rawpy_grayscale(sample.raw_filepath)
        jpeg_img = cv2.imread(sample.jpeg_filepath, 0)
        cv2.imwrite("steps/1_1_jpeg.jpg", jpeg_img)
        cv2.imwrite("steps/1_1_raw.jpg", raw_img)
        ''' 4.2.&3. Equalize and match histograms. '''
        jpeg_img = equalize_hist(jpeg_img)
        raw_img = match_histograms(raw_img, jpeg_img)
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
                matches = flann.knnMatch(des1,des2,k=2)
                good = []
                for m,n in matches:
                    if m.distance < 0.5*n.distance:
                        good.append(m)
                if len(good)>MIN_MATCH_COUNT:
                    src_pts = np.float32([ (kp1[m.queryIdx].pt[0] + x, kp1[m.queryIdx].pt[1] + y) for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ (kp2[m.queryIdx].pt[0] + x, kp2[m.queryIdx].pt[1] + y) for m in good ]).reshape(-1,1,2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    aligner.mappings[mapping_idx] = M
                else:
                    log_msg(f"[WARNING] {sample.file_name} patch {mapping_idx} ignored, (not enough good matches)")
                    M, mask = None, None
                mapping_idx +=1
                aligners.append(aligner)
        # cv2.imwrite("0.png",jpeg_patch)
        # cv2.imwrite("1.png",raw_patch)
    raw_preprocessing_transforms = Transforms()
    raw_preprocessing_transforms.log(np.diag([raw_sx, raw_sy]), "raw", desc=None)
    aggregated_aligner = Aligner(H=dataset.jpeg_h, W=dataset.jpeg_w, window_size=dataset.window_size, stride=dataset.stride, \
        addresses_LUT=cached_addresses_LUT, raw_preprocessing_transforms=raw_preprocessing_transforms)
    for mapping_idx in aggregated_aligner.mappings.keys():
        mappings = []
        for aligner in aligners:
            M = aligner.mappings[mapping_idx]
            if not isinstance(M,type(None)):
                mappings.append(M)
        avg_mapping = np.mean(mappings, axis=0)
        aggregated_aligner.mappings[mapping_idx] = avg_mapping
    rgb_raw = read_rawpy_rgb(sample.raw_filepath)
    result = draw_anno.annotate(rgb_raw, dataset.get_anno()[sample.file_name], transform_fnc=aggregated_aligner.apply, draw_org=True)
    cv2.imwrite("result.jpg", result)
    cv2.imwrite("raw.jpg", raw_img)
    cv2.imwrite("jpeg.jpg", jpeg_img)
    for (x,y) in [(0,0), (3936//2, 2624//2), (3936-1, 2624-1)]:
        print(x,y,"->",aggregated_aligner.apply(x,y))
    print("Done.")
    