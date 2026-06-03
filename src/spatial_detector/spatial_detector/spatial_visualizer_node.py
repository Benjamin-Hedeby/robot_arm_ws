#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge, CvBridgeError
import cv2

LABELS = [
    "ALOMY","ANGAR","APESV","ARTVU","AVEFA","BROST","BRSNN",
    "CAPBP","CENCY","CHEAL","CHYSE","CIRAR","CONAR","EPHHE",
    "EPHPE","EROCI","FUMOF","GALAP","GERMO","LAPCO","LOLMU",
    "LYCAR","MATCH","MATIN","MELNO","MYOAR","PAPRH","PLALA",
    "PLAMA","POAAN","POLAV","POLCO","POLLA","POLPE","RUMCR",
    "SENVU","SINAR","SOLNI","SONAS","SONOL","STEME","THLAR",
    "URTUR","VERAR","VERPE","VICHI","VIOAR"
]

class SpatialVisualizer(Node):
    def __init__(self):
        super().__init__('spatial_visualizer_node')
        self.bridge = CvBridge()
        self.latest_image = None

        # Publisher for annotated frames
        self.annotated_pub = self.create_publisher(
            Image, 'spatial_detector/image_annotated', 1)

        # 1. Subscribe to the raw RGB image
        self.create_subscription(
            Image,
            '/oak/rgb/image_rect',
            self.image_cb,
            1)

        # 2. Subscribe to the standard ROS 3D Detections
        self.create_subscription(
            Detection3DArray,
            '/oak/nn/spatial_detections',
            self.detections_cb,
            1)

        self.get_logger().info('Spatial Visualizer started! Waiting for images and detections...')

    def image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Image CB error: {e}')

    def detections_cb(self, msg: Detection3DArray):
        # Don't try to draw if we don't have an image yet
        if self.latest_image is None:
            return

        img = self.latest_image.copy()
        
        # 1. Get the actual dimensions of the video feed (e.g., 1080x1920)
        img_h, img_w = img.shape[:2] 
        nn_size = 640.0 # From your config's i_preview_width
        
        # 2. Calculate the "Center Crop & Scale" reverse mapping
        # 1080 / 640 = 1.6875 scale factor
        scale = img_h / nn_size 
        # (1920 - 1080) / 2 = 420 pixel offset to re-center the crop
        x_offset = (img_w - img_h) / 2.0 

        for det in msg.detections:
            # Extract the raw NN Pixel Coordinates (640x640 space)
            raw_x = det.bbox.center.position.x
            raw_y = det.bbox.center.position.y
            raw_w = det.bbox.size.x
            raw_h = det.bbox.size.y

            # 3. Apply the mapping to push them into 1080p space
            mapped_x = (raw_x * scale) + x_offset
            mapped_y = (raw_y * scale)
            mapped_w = raw_w * scale
            mapped_h = raw_h * scale

            # Calculate the top-left and bottom-right corners for cv2
            x1 = int(mapped_x - mapped_w / 2)
            y1 = int(mapped_y - mapped_h / 2)
            x2 = int(mapped_x + mapped_w / 2)
            y2 = int(mapped_y + mapped_h / 2)

            z_m = det.results[0].pose.pose.position.z

            # Draw the Bounding Box and Center Dot
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (int(mapped_x), int(mapped_y)), 5, (0, 0, 255), -1)

            # Decode the Label ID
            cid_str = det.results[0].hypothesis.class_id
            try:
                cid = int(cid_str)
                label = LABELS[cid] if 0 <= cid < len(LABELS) else str(cid)
            except ValueError:
                label = cid_str 

            # Add Text (Label and Depth in meters)
            text = f'{label}: {z_m:.2f} m'
            cv2.putText(
                img, text,
                (max(0, x1), max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish the final annotated image
        try:
            out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            out.header = msg.header
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