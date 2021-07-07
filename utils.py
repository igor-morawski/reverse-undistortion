import json
import os 
import os.path as op

CACHE_DIR = "cached"
CONFIGS_DIR = "configs"

def read_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data



def log_msg(msg, log=True):
    print(msg)
    if log:
        with open('align.log', 'a+') as f:
            f.write(msg+"\n")
    return True
