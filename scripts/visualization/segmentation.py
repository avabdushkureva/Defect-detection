import json
import cv2
import shapely
from shapely.geometry import Polygon
import numpy as np


def open_json_markup(path_to_json):
    with open(path_to_json) as f:
        s = json.load(f)
    return s


path = './datasets/AISS-CV/Rohbilder/Images Round 1/Joel/labels_joel_json.json'
dict_json = open_json_markup(path_to_json=path)

output_dir = Path('./datasets/AISS-CV/Rohbilder/Images Round 1/Joel/images_segmented')
output_dir.mkdir(parents=True, exist_ok=True)

label2color = {'box': (255, 255, 255),
               'wet': (255, 255, 0),
               'dent': (255, 0, 0),
               'hole': (0, 255, 255),
               'open': (0, 0, 255)}

for image in dict_json.keys():
    polygons = []
    labels = []
    for region in dict_json[image]['regions']:
        labels.append(region['region_attributes']['label'])
        if region['shape_attributes']['name'] == 'rect':
            x_left, y_left = region['shape_attributes']['x'], region['shape_attributes']['y']
            width, height = region['shape_attributes']['width'], region['shape_attributes']['height']
            x1, y1 = x_left, y_left
            x2, y2 = x_left, y_left + height
            x3, y3 = x_left + width, y_left + height
            x4, y4 = x_left + width, y_left
            polygons.append(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        else:
            poly = list(map(list, zip(region['shape_attributes']['all_points_x'],
                                      region['shape_attributes']['all_points_y'])))
            polygons.append(np.array(poly))

    img_name = Path('./datasets/AISS-CV/Rohbilder/Images Round 1/Joel/Images') / dict_json[image]['filename']
    img = cv2.imread(str(img_name))
    # mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    overlay = img.copy()

    for i in range(len(polygons)):
        cv2.polylines(
            img, pts=[polygons[i].reshape((-1, 1, 2))], isClosed=True, color=label2color[labels[i]], thickness=2
        )
        # cv2.fillPoly(mask, pts=[polygons[i]], color=label2color[labels[i]])
        # cv2.fillPoly(img, pts=[polygons[i]], color=label2color[labels[i]])

    alpha = 0.7
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    output_path = output_dir / dict_json[image]['filename']
    cv2.imwrite(str(output_path), img)
