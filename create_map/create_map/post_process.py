import os.path as op
from glob import glob
import cv2
import numpy as np

import config as cfg
from show_imgs import show_imgs


def main():
    maps = load_maps()
    visualize_maps(maps, "raw maps", 1.5)
    new_maps = apply_process(maps, post_process)
    visualize_maps(new_maps, "post maps", 1.5)
    # save_maps(new_imgs)


def load_maps():
    label_names = {label["id"]: label["name"] for label in cfg.SEGLABELCOLOR}
    maps = {}
    for id, name in label_names.items():
        file = op.join(cfg.RESULT_PATH, "1216_depth_lidar_v1", f"label_count_{id}.png")
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        maps[name] = img
        print("load image:", id, name, img.shape, np.max(img))
    return maps


def visualize_maps(maps, title, zoom):
    images = apply_process(maps, lambda x: prepare_visualize(x, zoom))
    show_imgs(images, title, 6)


def apply_process(maps, f):
    result = {key: f(val) for key, val in maps.items()}
    return result


def prepare_visualize(map, zoom):
    height, width = map.shape
    t, map = cv2.threshold(map, 2, 255, cv2.THRESH_BINARY)
    map = cv2.resize(map, (int(width * zoom), int(height * zoom)), cv2.INTER_NEAREST)
    return map


def post_process(map):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # map = cv2.GaussianBlur(map, (3, 3), 0)
    map = custom_median(map)
    # map = cv2.dilate(map, kernel)
    # map = cv2.medianBlur(map, 3)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # map = cv2.morphologyEx(map, cv2.MORPH_OPEN, kernel)
    return map


def custom_median(map):
    height, width = map.shape
    map_pad = cv2.copyMakeBorder(map, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    maps = []
    for y in range(3):
        for x in range(3):
            maps.append(map_pad[y:y+height, x:x+width])
    maps = np.stack(maps, axis=-1)
    maps = np.sort(maps, axis=-1)
    new_map = maps[:, :, -3]
    return new_map


if __name__ == "__main__":
    main()

