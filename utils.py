import json
import os 
import os.path as op

def read_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data