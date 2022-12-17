import numpy as np

RESULT_PATH = "/home/ri/colcon_ws/src/lidar2map/data"

NUM_CLASS = 1
MAX_DEPTH = 3.5
LOWER_HEIGHT = 0.2
IR_TO_ROBOT = np.array([[ 0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [ 0,-1, 0, 0.2],
                        [0, 0, 0, 1]])

IR2RGB = np.array([[0.9986, 0.0349, 0.0401, -0.067],
                   [-0.0396, 0.9916, 0.1227, 0.149],
                   [-0.0354, -0.1241, 0.9916, -0.080],
                   [0, 0, 0, 1]])

SCH_INSTRINSIC = [848, 480, 418.874, 418.874, 427.171, 239.457]

LGE_INSTRINSIC = [424, 240, 213.0231, 213.0231, 213.6875, 116.998]

GRID_RESOLUTION = 0.05
GRID_ORIGIN = np.array( [-3.3, 4, 0])

GRID_TYPE_NONE = np.asarray((205, 205, 205))
GRID_TYPE_FLOOR = np.asarray((254, 254, 254))
GRID_TYPE_WALL = np.asarray((0, 0, 0))

SEGLABELCOLOR = [{
    "name": "Wall",
    "id": 1,
    "color": [0.7, 0.8, 0.8],
    "trainId": 0
    },
    {
    "name": "Obstacle",
    "id": 2,
    "color": [0.2, 0.2, 0.2],
    "trainId": 1
    },
    {
    "name": "Floor",
    "id": 3,
    "color": [0.5, 0.5, 0],
    "trainId": 2
    },
    {
    "name": "Window_sill",
    "id": 4,
    "color": [0.8, 0.4, 0],
    "trainId": 3
    },
    {
    "name": "Ceiling",
    "id": 5,
    "color": [0.6, 0.6, 0.6],
    "trainId": 4
    },
    {
    "name": "Person",
    "id": 6,
    "color": [1, 0.878, 0.058],
    "trainId": 5
    },
    {
    "name": "Door",
    "id": 7 ,
    "color": [0.8, 0.9, 0.4],
    "trainId": 6
    },
    {
    "name": "Furniture",
    "id": 8,
    "color": [0.4, 0.3, 0],
    "trainId": 7
    },
    {
    "name": "Electronics",
    "id": 9,
    "color": [0, 1, 1],
    "trainId": 8
    },
    {
    "name": "Furniture_wall",
    "id": 10,
    "color": [0.3, 0.5, 0.5],
    "trainId": 9
    },
    {
    "name": "Window",
    "id": 11,
    "color": [0, 0.8, 0.8],
    "trainId": 10
    }
]

CTGR = ["None"] + [label["name"] for label in SEGLABELCOLOR]
CTGR_COLOR = [[0, 0, 0]] + [label["color"] for label in SEGLABELCOLOR]
CTGR_HEIGHT = [1, 1, 0.7, 0.7, 1.5, 1.5, 2, 2, 1, 2, 1.5, 1.5, 1]

