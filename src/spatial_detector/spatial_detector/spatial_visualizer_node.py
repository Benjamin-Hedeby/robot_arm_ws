#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from depthai_ros_msgs.msg import SpatialDetectionArray
from cv_bridge import CvBridge, CvBridgeError
import cv2

class SpatialVisualizer(Node):
    def __init__(self):
        super().__init__('spatial_visualizer_node')
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_header = None

        # Publisher for annotated frames
        self.annotated_pub = self.create_publisher(
            Image, 'spatial_detector/image_annotated', 1)

        # Subscribe to the raw RGB image
        self.create_subscription(
            Image,
            '/oak/rgb/image_rect',
            self.image_cb,
            1)

        # Subscribe to DepthAI's SpatialDetectionArray
        self.create_subscription(
            SpatialDetectionArray,
            '/oak/nn/spatial_detections',
            self.detections_cb,
            1)

        self.get_logger().info(
            'Spatial Visualizer started—waiting for /oak/rgb/image_rect & /oak/nn/spatial_detections'
        )

    def image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
            self.latest_header = msg.header
        except CvBridgeError as e:
            self.get_logger().error(f'Image CB error: {e}')

    def detections_cb(self, msg: SpatialDetectionArray):
        if self.latest_image is None:
            return

        img = self.latest_image.copy()
        for det in msg.detections:
            # Draw box from ROI
            roi = det.roi
            x1 = int(roi.x_offset)
            y1 = int(roi.y_offset)
            x2 = x1 + int(roi.width)
            y2 = y1 + int(roi.height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label & depth
            label = det.label if hasattr(det, 'label') else str(det.label_id)
            depth_m = det.spatial_coordinates.z
            text = f'{label}: {depth_m:.2f} m'
            cv2.putText(
                img, text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        try:
            out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            out.header = self.latest_header
            self.annotated_pub.publish(out)
        except CvBridgeError as e:
            self.get_logger().error(f'Publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SpatialVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
