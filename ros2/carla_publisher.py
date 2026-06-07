import rclpy

from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


class CarlaPublisher(Node):

    def __init__(self):

        super().__init__('carla_pub')

        self.pub = self.create_publisher(
            PoseStamped,
            '/carla/vehicle_pose',
            10
        )

        self.timer = self.create_timer(
            0.05,
            self.publish_pose
        )

    def publish_pose(self):

        msg = PoseStamped()

        msg.pose.position.x = 10.0
        msg.pose.position.y = 20.0
        msg.pose.position.z = 0.0

        self.pub.publish(msg)


rclpy.init()

node = CarlaPublisher()

rclpy.spin(node)