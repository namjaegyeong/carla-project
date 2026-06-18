import rclpy

from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


class PoseSubscriber(Node):

    def __init__(self):

        super().__init__('pose_sub')

        self.create_subscription(
            PoseStamped,
            '/carla/vehicle_pose',
            self.callback,
            10
        )

    def callback(self, msg):

        print(
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        )