import os.path as op
from glob import glob
import cv2
import numpy as np
import open3d as o3d

import config as cfg
from show_imgs import show_imgs
from create_map.build_map import SemanticMapRenderer
from create_map.build_map import GridMapRenderer

def main():
    grid_map = get_grid_map()
    maps = load_maps()
    visualize_maps(maps, "raw maps", 1.5)
    class_map = get_class_map(maps)
    show_2d_map(class_map, grid_map, "raw")
    show_3d_map(class_map, grid_map, "raw")

    post_maps = apply_process(maps, post_process)
    visualize_maps(post_maps, "post maps", 1.5)
    class_map = get_class_map(post_maps)
    show_2d_map(class_map, grid_map, "post")
    show_3d_map(class_map, grid_map, "post")

    final_maps = apply_process(post_maps, lambda x: grid_filter(x, grid_map))
    visualize_maps(final_maps, "grid maps", 1.5)
    class_map = get_class_map(final_maps)
    show_2d_map(class_map, grid_map, "grid")
    show_3d_map(class_map, grid_map, "grid")


def get_grid_map():
    map = cv2.imread("/home/ri/bagfiles/1216/depth/map.pgm")
    return map

def load_maps():
    label_names = {label["id"]: label["name"] for label in cfg.SEGLABELCOLOR}
    maps = {}
    file = op.join(cfg.RESULT_PATH, "label_count_map.npy")
    label_count_map = np.load(file, 'r')
    print("label_count_map", label_count_map.shape)
    max_val = np.max(label_count_map) * 0.8
    label_count_map = np.minimum(label_count_map, max_val).astype(float)
    label_count_map = (label_count_map/max_val*255).astype(np.uint8)
    
    for id, name in label_names.items():
        maps[name] = label_count_map[..., id]
        print("load image:", id, name, maps[name].shape, np.max(maps[name]))
    return maps


def visualize_maps(maps, title, zoom):
    images = apply_process(maps, lambda x: prepare_visualize(x, zoom))
    show_imgs(images, title, 6, wait_key=10)


def get_class_map(map):
    none = np.zeros((map["Wall"].shape[0], map["Wall"].shape[1]))
    ctgr = list(map.values())
    ctgr.insert(0, none)
    ctgr = np.array(ctgr)
    ctgr = ctgr.transpose(1, 2, 0)
    return ctgr
    

def apply_process(maps, f):
    result = {key: f(val) for key, val in maps.items()}
    return result


def prepare_visualize(map, zoom):
    height, width = map.shape
    t, map = cv2.threshold(map, 50, 255, cv2.THRESH_BINARY)
    map = cv2.resize(map, (int(width * zoom), int(height * zoom)), cv2.INTER_NEAREST)
    return map


def post_process(map):
    map = custom_median(map)
    return map


def custom_median(map):
    height, width = map.shape
    map_pad = cv2.copyMakeBorder(map, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    maps = []
    for y in range(3):
        for x in range(3):
            maps.append(map_pad[y:y+height, x:x+width])
    maps = np.stack(maps, axis=-1)
    maps = np.sort(maps, axis=-1)
    new_map = maps[:, :, -3]
    return new_map


def grid_filter(label_map, grid_map):
    label_map[grid_map[..., 0] != 0] = 0
    return label_map


def show_2d_map(class_count, grid_map, style):
    class_id_map = np.argmax(class_count, axis=2)
    class_color_map = grid_map.copy()
    for i, color in enumerate(cfg.CTGR_COLOR):
        if i==0:
            continue
        class_color_map[class_id_map==i] = (np.array(color)*255).astype(np.uint8)[::-1]
    class_color_map = cv2.resize(class_color_map, (int(grid_map.shape[1]*3), int(grid_map.shape[0]*3)), cv2.INTER_NEAREST)
    cv2.imshow(style + " class map", class_color_map)
    cv2.imwrite(cfg.RESULT_PATH + "")
    cv2.waitKey(100)


def show_3d_map(layer_class_map, grid_gray_map, style):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0.1])
    grid_map = grid_gray_map.copy()
    grid_map[grid_map == 0] = 2
    grid_map[grid_map == 205] = 0
    grid_map[grid_map == 254] = 1
    class_map = np.argmax(layer_class_map, axis=2)
    class_mask = np.max(layer_class_map, axis=2) > 2
    class_map = class_map * class_mask
    grid_map = GridMapRenderer(grid_map[..., 0]).build
    total_map = SemanticMapRenderer(class_map).build
    o3d.visualization.draw_geometries([frame, grid_map, total_map])


if __name__ == "__main__":
    main()

