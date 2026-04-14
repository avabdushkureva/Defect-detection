import json
import cv2
import shapely
from shapely.geometry import Polygon


def open_json_markup(path_to_json):
    with open(path_to_json) as f:
        s = json.load(f)
    return s


path = './AISS-CV/Rohbilder/Images Round 1/Joel/labels_joel_json.json'
dict_json = open_json_markup(path_to_json=path)

output_dir = Path('./AISS-CV/Rohbilder/Images Round 1/Joel/images_markup')
output_dir.mkdir(parents=True, exist_ok=True)

for image in dict_json.keys():
    bboxes = []
    for region in dict_json[image]['regions']:
        if region['shape_attributes']['name'] == 'rect':
            bboxes.append([region['shape_attributes']['x'], region['shape_attributes']['y'],
                          region['shape_attributes']['width'], region['shape_attributes']['height']])
        else:
            poly = Polygon(list(map(list, zip(region['shape_attributes']['all_points_x'],
                                              region['shape_attributes']['all_points_y']))))
            x_left, y_left, x_right, y_right = poly.bounds
            width, height = x_right - x_left, y_right - y_left
            bboxes.append([x_left, y_left, width, height])

    img_name = Path('./AISS-CV/Rohbilder/Images Round 1/Joel/Images') / dict_json[image]['filename']
    img = cv2.imread(str(img_name))

    for bbox in bboxes:
        x_left, y_left, width, height = bbox
        x_right, y_right = x_left + width, y_left + height
        img = cv2.rectangle(
            img, (int(x_left), int(y_left)), (int(x_right), int(y_right)), color=(0, 255, 0), thickness=3
        )

    output_path = output_dir / dict_json[image]['filename']
    cv2.imwrite(str(output_path), img)
