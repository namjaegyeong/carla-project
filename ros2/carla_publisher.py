import numpy as np

from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CarlaPublisher(Node):

    def __init__(self):

        super().__init__('carla_pub')

        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

    def publish_carla_image(self, carla_image):

        img = np.frombuffer(
            carla_image.raw_data,
            dtype=np.uint8
        ).reshape(
            carla_image.height,
            carla_image.width,
            4
        )

        img = img[:, :, :3]

        msg = self.bridge.cv2_to_imgmsg(
            img,
            encoding='bgr8'
        )

        self.image_pub.publish(msg)