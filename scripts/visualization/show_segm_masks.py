from pathlib import Path
import cv2
import numpy as np
import random


def main():
    path_to_img = "./datasets/Second_stage/Dent/images/train/IMG_3856_frame_68_object_0.jpg"
    path_to_txt = "./datasets/Second_stage/Dent/labels/train/IMG_3856_frame_68_object_0.txt"
    
    image = cv2.imread(path_to_img)
    overlay = image.copy()
    img_h, img_w, _ = image.shape

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    with open(path_to_txt, "r") as txt_file:
        lines = txt_file.readlines()
        lines = [i.strip().split(" ") for i in lines]
        polygon = [i[1:] for i in lines]
        colors = np.linspace(64, 255, len(polygon))
        random_colors = [(random.randint(1, 255), int(random.randint(1, 255)), random.randint(1, 255))
                         for _ in range(len(polygon))]

        for poly, color_value, random_color in zip(polygon, colors, random_colors):
            poly = np.array(poly).astype(np.float32).reshape(-1, 2)
            poly[:, 0] *= img_w
            poly[:, 1] *= img_h
            cv2.fillPoly(mask, pts=[poly.astype(np.int32)],
                         color=(int(color_value), int(color_value), int(color_value)))
            cv2.fillPoly(image, pts=[poly.astype(np.int32)], color=random_color)

        cv2.addWeighted(overlay, 0.5, image, 1 - 0.5, 0, image)


main()
