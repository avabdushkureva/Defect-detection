import copy
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
from natsort import natsorted
from tqdm import tqdm
import shutil


def main():
    src_dir = "./data/AISS-CV_images_masks"
    dst_dir = "./data/AISS-CV-yolov8-prepared"

    Path(dst_dir).mkdir(exist_ok=True)

    dst_dir_images = Path(dst_dir, "images")
    dst_dir_labels = Path(dst_dir, "labels")
    dst_dir_images.mkdir(exist_ok=True)
    dst_dir_labels.mkdir(exist_ok=True)

    paths_to_images = natsorted(list(Path(src_dir).rglob("*.jpg")))
    paths_to_masks = natsorted(list(Path(src_dir).rglob("*.png")))

    for pt_img, pt_mask in tqdm(zip(paths_to_images, paths_to_masks), total=len(paths_to_masks)):
        mask = cv2.imread(str(pt_mask), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape

        polygons: list[np.ndarray] = []
        for color_value in np.unique(mask)[np.unique(mask) > 0]:
            mask_per_object = np.where(mask == color_value, mask, 0)
            contours, _ = cv2.findContours(mask_per_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            best_polygon: np.ndarray
            best_shape = 0
            for contour in contours:
                if contour.shape[0] > best_shape:
                    best_polygon = np.squeeze(contour)
                    best_shape = contour.shape[0]

            # zero_mask = np.zeros((mask_per_object.shape[0], mask_per_object.shape[1], 3), dtype=np.uint8)
            # cv2.fillPoly(zero_mask, pts=[best_polygon], color=(255, 255, 255))
            polygon_shapely = Polygon(best_polygon)
            polygon_simplified = polygon_shapely.simplify(tolerance=0.8)
            polygon_simplified = np.array(polygon_simplified.exterior.coords.xy).T
            polygon_simplified /= [w, h]
            polygons.append(polygon_simplified)

        # path_to_txt_file = Path(dst_dir_labels, pt_img.name)
        path_to_txt_file = dst_dir_labels / (pt_img.stem + ".txt")
        with open(path_to_txt_file, "w") as txt_file:
            for polygon in polygons:
                str_id = [str(0)]
                str_poly = [str(i) for i in polygon.ravel().tolist()]
                final_result = str_id + str_poly
                final_str = " ".join(final_result)
                txt_file.write(final_str + "\n")

        shutil.copy(str(pt_img), str(dst_dir_images))


main()
