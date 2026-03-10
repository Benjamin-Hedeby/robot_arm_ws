#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseArray, Pose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class SpatialPublisher(Node):
    def __init__(self):
        super().__init__('spatial_publisher_node')

        # QoS: best-effort, keep last 5
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.pose_pub = self.create_publisher(
            PoseArray,
            'spatial_detector/detected_poses',
            qos)

        # Subscribe to the raw Detection3DArray from the Oak-D
        self.create_subscription(
            Detection3DArray,
            '/oak/nn/spatial_detections',
            self.detections_cb,
            qos)

        self.get_logger().info(
            'Spatial Publisher started—listening on /oak/nn/spatial_detections'
        )

    def detections_cb(self, msg: Detection3DArray):
        pa = PoseArray()
        pa.header = msg.header

        for det in msg.detections:
            if not det.results:
                continue
            # Extract 3D position
            pose = Pose()
            pose.position = det.results[0].pose.pose.position
            pose.orientation.w = 1.0
            pa.poses.append(pose)

        if pa.poses:
            self.pose_pub.publish(pa)
            self.get_logger().debug(
                f'Published {len(pa.poses)} poses')


def main(args=None):
    rclpy.init(args=args)
    node = SpatialPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
