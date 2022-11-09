import rclpy
import numpy as np
import cv2
# import config as cfg
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray


class MapPublisher(Node):
    def __init__(self, map_image):
        super().__init__('map_pub')
        self.map_publisher = self.create_publisher(Int16MultiArray, 'grid_map', 10)
        self.image = cv2.imread(map_image)
        self.timer = self.create_timer(1, self.publish_map)

    def publish_map(self):
        y, x, c = self.image.shape
        map = Int16MultiArray()
        base = np.zeros((y, x)).astype(int)
        ctgr = [np.asarray((205, 205, 205)),  np.asarray((254, 254, 254)), np.asarray((0, 0, 0))]
        # ctgr = [cfg.GRID_TYPE_NONE, cfg.GRID_TYPE_FLOOR, cfg.GRID_TYPE_WALL]

        for i, c in enumerate(ctgr):
            x_cord, y_cord, z_cord = np.where(self.image == c)
            for (x_, y_) in zip(x_cord, y_cord):
                base[x_][y_] = i
        # print(type(base))
        base = base.reshape(1, -1).tolist()
        map.data = base[0] + [y, x]
        self.map_publisher.publish(map)

def main(args=None):
    rclpy.init(args=args)
    image = "/home/ri/bagfiles/datasets/cheonan/221026/map/first_date-second.pgm"
    node = MapPublisher(image)
    rclpy.spin(node)

    
if __name__ == "__main__":
    main()