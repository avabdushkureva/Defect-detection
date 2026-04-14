import json
import cv2
import numpy as np
import shutil
from pathlib import Path


def open_json_markup(path_to_json):
    with open(path_to_json) as f:
        s = json.load(f)
    return s


src_dir = './datasets/AISS-CV/Rohbilder/Images Round 1/Joel'
src_dir_images = Path(src_dir) / 'Images'
dict_json_path = Path(src_dir) / 'labels_joel_json.json'
dict_json = open_json_markup(path_to_json=dict_json_path)

label2id = {'wet': 0,
            'dent': 1,
            'hole': 2,
            'open': 3}

dst_dir = './datasets/AISS-CV_defects_yolov8/Joel'
dst_dir_images = Path(dst_dir) / 'images'
dst_dir_labels = Path(dst_dir) / 'labels'

dst_dir_images.mkdir(parents=True, exist_ok=True)
dst_dir_labels.mkdir(parents=True, exist_ok=True)

for image in dict_json.keys():
    img_path = src_dir_images / dict_json[image]["filename"]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    segmentations = []
    labels = []

    for region in dict_json[image]['regions']:
        if region['shape_attributes']['name'] == 'rect':
            continue
        else:
            poly = list(
                map(list, zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))
            )
            poly = np.array(poly, dtype=float)
            poly /= [w, h]
            segmentations.append(np.array(poly))
        label = region['region_attributes']['label']
        labels.append(label2id[label])

    if not segmentations:
        continue

    txt_filename = Path(dict_json[image]["filename"]).stem + '.txt'
    path_to_txt_file = dst_dir_labels / txt_filename

    with open(path_to_txt_file, "w") as txt_file:
        for i in range(len(segmentations)):
            str_id = [str(labels[i])]
            str_poly = [str(j) for j in segmentations[i].ravel().tolist()]
            final_result = str_id + str_poly
            final_str = " ".join(final_result)
            txt_file.write(final_str + "\n")

    shutil.copy(str(img_path), str(dst_dir_images))
