import glob 
import os
import os.path as op
import sys
import exifread
import utils
import tqdm
import json
from collections import Counter

CONFIG_REQUIRED_FIELDS = ["name", "data_root", "jpeg_h", "jpeg_w", "raw_h", "raw_h", "jpeg_anno", "jpeg_ext", "raw_ext"]
SUPPORTED_ANNO_EXT = ["csv"]
ACCEPTED_LINE_LENGTHS = [5, 6]
ANNO_FIELD_SEP = ","

class Sample_From_File_Name:
    def __init__(self, file_name, data_root, jpeg_ext, raw_ext):
        self.jpeg_filepath = op.join(data_root,file_name.split(".")[0]+"."+jpeg_ext)
        self.raw_filepath = op.join(data_root,file_name.split(".")[0]+"."+raw_ext)
        for f in (self.jpeg_filepath, self.raw_filepath):
            if not op.exists(f): raise Exception(f"{f} not found.")
        with open(self.jpeg_filepath, 'rb') as f:
            tags = exifread.process_file(f) 
        assert tags['EXIF FocalLength']
        self.focal_length = eval(tags['EXIF FocalLength'].printable)
        
class Sample:
    def __init__(self, jpeg_filepath, raw_filepath, focal_length):
        self.jpeg_filepath = jpeg_filepath
        self.raw_filepath = raw_filepath
        self.focal_length = focal_length

class Dataset:
    def __init__(self, config, caching=True):
        for field in CONFIG_REQUIRED_FIELDS:
            object.__setattr__(self, field, config[field])
        self.file_names = self._parse_filelist(self.jpeg_anno)
        self.samples = []
        self.caching = caching 
        if not op.exists(utils.CACHE_DIR):
            os.mkdir(utils.CACHE_DIR)
        
        self.cached_dataset = op.join(utils.CACHE_DIR, self.name+".json")
        if (not self.caching) or (self.caching and not op.exists(self.cached_dataset)):
            if self.caching:
                cached_data = {}
            for file_name in tqdm.tqdm(self.file_names):
                sample = Sample_From_File_Name(file_name, self.data_root, self.jpeg_ext, self.raw_ext)
                self.samples.append(sample)
                if self.caching:
                    cached_data[file_name] = {"jpeg_filepath" : sample.jpeg_filepath, 
                    "raw_filepath" : sample.raw_filepath, 
                    "focal_length" : sample.focal_length}
            if self.caching:
                with open(self.cached_dataset, "w") as f:
                    json.dump(cached_data, f)
        else:
            cached_data = utils.read_json(self.cached_dataset)
            for sample_args in cached_data.values():
                sample = Sample(**sample_args)
                self.samples.append(sample)
        self._original_samples = self.samples.copy()
        self.focal_length = -1

    def _parse_filelist(self, filepath):
        ext = self.jpeg_anno.split(".")[-1].lower()
        if ext not in SUPPORTED_ANNO_EXT:
            raise ValueError(f"Supported annotation types: {SUPPORTED_ANNO_EXT}")
        if ext == "csv":
            file_names =  self._parse_filelist_csv(filepath)
        return file_names

    def _parse_filelist_csv(self, filepath):
        with self._open_for_csv(filepath) as f:
            anno = f.readlines()
        file_names = []
        for line in anno: 
            chunks = line.split(ANNO_FIELD_SEP)
            assert len(chunks) in ACCEPTED_LINE_LENGTHS
            file_name = chunks[0]
            if file_name not in file_names:
                file_names.append(file_name)
        return file_names

    def _open_for_csv(self, path):
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def filter_by_focal_length(self, focal_length):
        if focal_length > 0:
            self.samples = []
            for sample in self._original_samples:
                if sample.focal_length == focal_length:
                    self.samples.append(sample)
        elif focal_length == -1:
            self.samples = self._original_samples.copy()
        else:
            raise ValueError(f"Invalid focal lenght value ({focal_length}).")

    def show_stats(self):
        focal_lengths = [s.focal_length for s in self.samples]
        stats = Counter(focal_lengths)
        print(f"{self.name}")
        print("----------------")
        print(stats)
        print("----------------")
        print(f"Total: {len(focal_lengths)}/{len(self._original_samples)}")