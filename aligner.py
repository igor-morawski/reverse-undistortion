import os 
import os.path as op
import json
import numpy as np
import copy
import pickle

class Aligner:
    def __init__(self, H, W, window_size : int, stride, addresses_LUT = None, mappings = None, raw_preprocessing_transforms = None, log=False, load_from=None):
        ''' Maps are numbered in C-order (like np.flatten)'''
        self._req_attributes = ["H", "W", "window_size", "stride", "addresses_LUT", "mappings", "_rptc", "_maps_n"]
        self._att_to_serialize = {"addresses_LUT" : int, "mappings" : None, "_rptc" : None}
        assert isinstance(window_size, int)
        assert stride <= window_size
        assert not H % window_size 
        assert not W % window_size
        assert not H % stride 
        assert not W % stride

        self.log = log
        if load_from:
            _H = H
            _W = W
            _window_size = window_size
            _stride = stride
            self.load(load_from)
            b = [self.H != _H, self.W != W, self.window_size != _window_size, self.stride != _stride]
            if any(b):
                self._log_msg("Overriding aligner init.", log=self.log)
                print(self.H, _H, ";", self.W, W, ";", self.window_size, _window_size, ";", self.stride, _stride)
            return

        if not isinstance(raw_preprocessing_transforms, type(None)):
            #raw_preprocessing_transforms composed
            if isinstance(raw_preprocessing_transforms, Transforms):
                self._rptc = np.linalg.inv(raw_preprocessing_transforms.compose())
            elif isinstance(raw_preprocessing_transforms, np.ndarray):
                assert raw_preprocessing_transforms.shape == (3,3)
                self._rptc =  np.linalg.inv(raw_preprocessing_transforms)
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
        return
        
    
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
        if x==self.W:
            x-=1
        if y==self.H:
            y-=1
        mapping_idx = self.addresses_LUT[y, x] 
        mapping = self.mappings[mapping_idx]
        _result = mapping @ np.array([x, y, 1])
        _result = self._rptc @ _result
        if floor:
            _result = _result.astype(int)
        return (_result[0], _result[1])

    def save(self, filepath):
        data = {}
        for attribute in self._req_attributes:
            data[attribute] = getattr(self, attribute)
        np.save(filepath, data)
        return True

    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True).item()
        for req_key in self._req_attributes:
            if req_key not in data.keys(): 
                raise ValueError(f"Required key {req_key} not found in {filepath}!")
            setattr(self, req_key, data[req_key])
        self._log_msg(f"Loaded {filepath}", log=self.log)
        return True
        
    def _save_json(self, filepath):
        data = {}
        for attribute in self._req_attributes:
            data[attribute] = getattr(self, attribute)
            if attribute in self._att_to_serialize.keys():
                if isinstance(data[attribute], np.ndarray):
                    serialized = np.array(data[attribute]).copy().tolist()
                elif isinstance(data[attribute], dict):
                    serialized = {}
                    for k, v in copy.deepcopy(data[attribute]).items():
                        a = data[attribute][k]
                        serialized_a = np.array(a).tolist()
                        serialized[int(k)] = serialized_a
                else: 
                    raise ValueError
                data[attribute] = serialized
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        self._log_msg(f"Aligner saved to {filepath}", log=self.log)
        return True


    def _load_json(self, filepath):
        # XXX needs a fix << self.mappings.keys() are strings instead of integers!
        with open(filepath) as f:
            data = json.load(f)
        for req_key in self._req_attributes:
            if req_key not in data.keys(): 
                raise ValueError(f"Required key {req_key} not found in {filepath}!")
            setattr(self, req_key, data[req_key])
            if req_key in self._att_to_serialize.keys():
                 data[req_key] = np.array(data[req_key], dtype=self._att_to_serialize[req_key])
            #     data[req_key] = data[req_key].reshape([3,3], dtype=self._att_to_serialize[req_key])
        self._log_msg(f"Loaded {filepath}", log=self.log)
        return True
        
        

    def _log_msg(self, msg, log=True):
        print(msg)
        if log:
            with open('align.log', 'a+') as f:
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
