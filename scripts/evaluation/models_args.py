import yaml
from yaml import load
import numpy as np
from pathlib import Path


def open_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as f:
        s = yaml.safe_load(f)
    return s


src_dir = './runs/yolov8_box_parcel_second_stage'
paths_to_args = list(Path(src_dir).rglob("*.yaml"))

for pt_to_args in paths_to_args:
    args = open_yaml(path_to_yaml=str(pt_to_args))
    print(args['epochs'], args['batch'], args['imgsz'])
