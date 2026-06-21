import json
from std_msgs.msg import String

from rclpy.node import Node

class YoloSubscriber(Node):

    def __init__(self):

        super().__init__('yolo_sub')

        self.yolo_sub = self.create_subscription(
            String,
            "/yolo/result",
            self.result_callback,
            10
        )

    def result_callback(self, msg):

        detections = json.loads(msg.data)

        for det in detections:
            print(
                det["cls"],
                det["conf"],
                det["xyxy"]
            )