#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge, CvBridgeError
import cv2

# The labels from your overlay script
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
        
        # Camera Intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        # Publisher for annotated frames
        self.annotated_pub = self.create_publisher(
            Image, 'spatial_detector/image_annotated', 1)

        # 1. Subscribe to Camera Info (Needed for reverse-projection math)
        self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info',
            self.caminfo_cb,
            10)

        # 2. Subscribe to the raw RGB image
        self.create_subscription(
            Image,
            '/oak/rgb/image_rect',
            self.image_cb,
            1)

        # 3. Subscribe to the standard ROS 3D Detections
        self.create_subscription(
            Detection3DArray,
            '/oak/nn/spatial_detections',
            self.detections_cb,
            1)

        self.get_logger().info('Spatial Visualizer started—Waiting for camera feeds...')

    def caminfo_cb(self, msg: CameraInfo):
        # Extract focal lengths and center points
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Image CB error: {e}')

    def detections_cb(self, msg: Detection3DArray):
        # Don't try to draw if we don't have an image or camera intrinsics yet
        if self.latest_image is None or self.fx is None:
            return

        img = self.latest_image.copy()
        
        for det in msg.detections:
            # Grab the 3D center point
            p = det.results[0].pose.pose.position
            x_m, y_m, z_m = p.x, p.y, p.z
            
            # Ignore invalid depths
            if z_m <= 0.0:
                continue

            # --- THE REVERSE MATH ---
            # Project the 3D center point back into 2D pixel space
            u = int(self.fx * x_m / z_m + self.cx)
            v = int(self.fy * y_m / z_m + self.cy)

            # Project the physical 3D bounding box size into 2D pixel width/height
            size_x_m = det.bbox.size.x
            size_y_m = det.bbox.size.y
            
            # If the network gives us 3D sizes, project them. Otherwise default to a 50px box
            w_px = int((size_x_m / z_m) * self.fx) if size_x_m > 0 else 50
            h_px = int((size_y_m / z_m) * self.fy) if size_y_m > 0 else 50

            # Calculate the top-left and bottom-right corners
            x1 = int(u - w_px / 2)
            y1 = int(v - h_px / 2)
            x2 = int(u + w_px / 2)
            y2 = int(v + h_px / 2)

            # Draw the Bounding Box and Center Dot
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (u, v), 5, (0, 0, 255), -1)

            # Decode the Label ID
            cid_str = det.results[0].hypothesis.class_id
            try:
                cid = int(cid_str)
                label = LABELS[cid] if 0 <= cid < len(LABELS) else str(cid)
            except ValueError:
                label = cid_str # Fallback if class_id is already a string

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