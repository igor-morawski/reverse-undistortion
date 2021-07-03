import glob 
import os
import os.path as op

CONFIG_REQUIRED_FIELDS = ["data_root", "jpeg_h", "jpeg_w", "raw_h", "raw_h", "jpeg_anno"]
SUPPORTED_ANNO_EXT = ["csv"]
class Dataset:
    def __init__(self, config):
        for field in CONFIG_REQUIRED_FIELDS:
            object.__setattr__(self, field, config[field])
    
    def _parse_filelist(self, filepath):
        ext = self.jpeg_anno.split(".")[-1].lower()
        if ext not in SUPPORTED_ANNO_EXT:
            raise ValueError(f"Supported annotation types: {SUPPORTED_ANNO_EXT}")
        if ext == "csv":
            return _parse_filelist_csv

    def _parse_filelist_csv(self, filepath):
        with self._open_for_csv(filepath) as f:
            anno = f
    
    def _open_for_csv(self, path):
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')
