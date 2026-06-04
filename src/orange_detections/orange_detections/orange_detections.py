import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped  
from cv_bridge import CvBridge
import cv2
import numpy as np
from visualization_msgs.msg import Marker

class OrangeTrackerNode(Node):
    def __init__(self):
        super().__init__('orange_tracker_node')
        self.bridge = CvBridge()

        # 1. Store the latest depth image and camera matrix
        self.latest_depth_img = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # 2. Setup Subscribers
        self.depth_sub = self.create_subscription(Image, '/oak/stereo/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.info_callback, 10)
        self.rgb_sub = self.create_subscription(Image, '/oak/rgb/image_rect', self.rgb_callback, 10)

        # 3. Setup Publishers
        self.target_pub = self.create_publisher(PointStamped, '/orange_target_3d', 10) # <-- FIXED: PointStamped
        self.overlay_pub = self.create_publisher(Image, '/orange_tracker/detection_overlay', 10)
        self.marker_pub = self.create_publisher(Marker, '/orange_tracker/target_marker', 10)

        self.get_logger().info("Orange Tracker Node Started! Waiting for camera feeds...")

    def info_callback(self, msg):
        self.fx = msg.k[0]
        self.cx = msg.k[2]
        self.fy = msg.k[4]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg):
        if self.latest_depth_img is None or self.fx is None:
            return

        # Convert ROS Image to OpenCV Image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Black out the top 150 pixels to ignore the orange robot parts
        frame[0:170,:] = [0, 0, 0]

        frame[:, 0:300] = [0, 0, 0]

        # --- THE HSV COLOR FILTER ---
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 120, 100]) 
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

        # Clean up the mask (removes noise)
        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

        # Find the blobs
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_found = False
        valid_contours = []

        # Filter out tiny noise contours
        for cnt in contours:
            if cv2.contourArea(cnt) > 200: 
                valid_contours.append(cnt)

        # Process the largest valid target
        if valid_contours:
            largest_cnt = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_cnt)

            # --- 2D VISUALIZATION (Draw boxes on the image) ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            pixel_x = int(x + w / 2)
            pixel_y = int(y + h / 2)
            cv2.circle(frame, (pixel_x, pixel_y), 8, (255, 0, 0), -1)
            cv2.putText(frame, "TARGET DETECTED", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            target_found = True

            # --- 3D MATH & MARKERS ---
            depth_mm = self.latest_depth_img[pixel_y, pixel_x]
            if depth_mm > 0: 
                z_meters = depth_mm / 1000.0
                x_meters = (pixel_x - self.cx) * z_meters / self.fx
                y_meters = (pixel_y - self.cy) * z_meters / self.fy

                # 1. Publish the 3D Point
                target_msg = PointStamped()
                target_msg.header.stamp = self.get_clock().now().to_msg()
                target_msg.header.frame_id = "oak_rgb_camera_optical_frame" 
                target_msg.point.x = float(x_meters)
                target_msg.point.y = float(y_meters)
                target_msg.point.z = float(z_meters)
                self.target_pub.publish(target_msg)
                
                # Add 3D coordinates text to the video
                info_text = f"Coord: ({x_meters:.2f}, {y_meters:.2f}, {z_meters:.2f})m"
                cv2.putText(frame, info_text, (x, y + h + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 2. Publish the 3D Sphere Marker for RViz
                marker = Marker()
                marker.header.stamp = target_msg.header.stamp
                marker.header.frame_id = "oak_rgb_camera_optical_frame"
                marker.ns = "weeds"
                marker.id = 0
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(x_meters)
                marker.pose.position.y = float(y_meters)
                marker.pose.position.z = float(z_meters)
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.r = 1.0
                marker.color.g = 0.3
                marker.color.b = 0.0
                marker.color.a = 0.8
                self.marker_pub.publish(marker)

        if not target_found:
             cv2.putText(frame, "WAITING FOR TARGET...", (100, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        # --- ALWAYS PUBLISH THE 2D OVERLAY ---
        # (This must be outside all loops so the camera feed never freezes in RViz)
        overlay_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.overlay_pub.publish(overlay_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OrangeTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Orange Tracker Node...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()