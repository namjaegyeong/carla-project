import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageRawPublisher(Node):

    def __init__(self):

        super().__init__('image_raw_pub')

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

        # BGRA -> BGR
        img = img[:, :, :3]

        msg = Image()

        msg.height = carla_image.height
        msg.width = carla_image.width

        msg.encoding = "bgr8"
        msg.is_bigendian = False

        msg.step = carla_image.width * 3

        msg.data = img.tobytes()

        self.image_pub.publish(msg)