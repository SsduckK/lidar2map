import numpy as np

CTGR = ["Wall", "Obstacle", "Floor", "Window", "Ceiling", "Person", "Door"]
CTGR_HEIGHT = [0.5, 0.1, 3, 0.6, 1, 2, 1]
CTGR_COLOR = [[0.5, 0, 0.5], [1, 1, 1], [0, 0.5, 0.5], [0.6, 0.4, 0], [0.6, 0.6, 0.6], [0.058, 0.878, 1], [0.9, 0.9, 0.8]]

NUM_CLASS = 1
MAX_DEPTH = 3.5
LOWER_HEIGHT = 0.2
CAM_TO_BASE = np.array([[ 0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [ 0,-1, 0, 0.2],
                        [0, 0, 0, 1]])

IR2RGB = np.array([[0.9986, 0.0349, 0.0401, -0.067],
                   [-0.0396, 0.9916, 0.1227, 0.149],
                   [-0.0354, -0.1241, 0.9916, -0.080],
                   [0, 0, 0, 1]])

SCH_INSTRINSIC = np.array([848, 480, 418.874, 418.874, 427.171, 239.457])

LGE_INSTRINSIC = np.array([424, 240, 213.0231, 213.0231, 213.6875, 116.998])

GRID_RESOLUTION = 0.05
GRID_ORIGIN = np.array([-9.77, -4.86, 0])

GRID_TYPE_NONE = np.asarray((205, 205, 205))
GRID_TYPE_FLOOR = np.asarray((254, 254, 254))
GRID_TYPE_WALL = np.asarray((0, 0, 0))

SEGLABELCOLOR = [{
    "name": "Wall",
    "id": 1,
    "color": [
    0.466666666666667,
    0.6745,
    0.188235
    ],
    "trainId": 0
},
{
    "name": "Obstacle",
    "id": 2,
    "color": [
        0.85,
        0.325,
        0.098
    ],
    "trainId": 1
    },
    {
    "name": "Floor",
    "id": 3,
    "color": [
        0,
        0.44705,
        0.7411
    ],
    "trainId": 2
    },
    {
    "name": "Window",
    "id": 4,
    "color": [
        0.63529,
        0.07843,
        0.18431
    ],
    "trainId": 3
    },
    {
    "name": "Ceiling",
    "id": 5,
    "color": [
        0.45882,
        0.00784,
        0.45882
    ],
    "trainId": 4
    },
    {
    "name": "Person",
    "id": 6,
    "color": [
        0,
        0.7607,
        0.3529
    ],
    "trainId": 5
    },
    {
    "name": "Door",
    "id": 7 ,
    "color": [
        1,
        0,
        0
    ],
    "trainId": 6
}]
