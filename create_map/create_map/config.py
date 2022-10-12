import numpy as np

NUM_CLASS = 1
MAX_DEPTH = 3.5
LOWER_HEIGHT = 0.2
CAM_TO_BASE = np.array([[ 0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [ 0,-1, 0, 0.2],
                        [0, 0, 0, 1]])

GRID_RESOLUTION = 0.05
GRID_ORIGIN = np.array([-3.53, 2.9, 0])
