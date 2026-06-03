import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped  
from std_msgs.msg import Empty  # Used for our trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
from visualization_msgs.msg import Marker

class TriggeredOrangeTracker(Node):
    def __init__(self):
        super().__init__('orange_tracker_node')
        self.bridge = CvBridge()

        # --- AVERAGING & TRIGGER STATE ---
        self.is_collecting = False
        self.measurement_buffer = []
        self.samples_needed = 20  # At 30fps, this collects 1 second of data

        # Camera Intrinsic Variables
        self.latest_depth_img = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Setup Subscribers
        self.depth_sub = self.create_subscription(Image, '/oak/stereo/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.info_callback, 10)
        self.rgb_sub = self.create_subscription(Image, '/oak/rgb/image_rect', self.rgb_callback, 10)
        
        # --- THE NEW TRIGGER SUBSCRIBER ---
        self.trigger_sub = self.create_subscription(Empty, '/trigger_detection', self.trigger_callback, 10)

        # Setup Publishers
        self.target_pub = self.create_publisher(PointStamped, '/orange_target_3d', 10) 
        self.overlay_pub = self.create_publisher(Image, '/orange_tracker/detection_overlay', 10)
        self.marker_pub = self.create_publisher(Marker, '/orange_tracker/target_marker', 10)

        self.get_logger().info("Node Started! Waiting for trigger on /trigger_detection...")

    def trigger_callback(self, msg):
        if not self.is_collecting:
            self.get_logger().info(f"Trigger received! Collecting {self.samples_needed} samples...")
            self.measurement_buffer = []  # Clear old data
            self.is_collecting = True     # Start the collection engine

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

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame[0:160, :] = [0, 0, 0] # Mask robot parts

        # HSV Filter
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([0, 90, 70]) 
        upper_orange = np.array([20, 255, 255])
        orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        target_found = False
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

        if valid_contours:
            largest_cnt = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_cnt)

            pixel_x = int(x + w / 2)
            pixel_y = int(y + h / 2)
            target_found = True

            # 2D Visualization
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(frame, (pixel_x, pixel_y), 8, (255, 0, 0), -1)

            # --- ROBUST DEPTH CALCULATION (ROI) ---
            box_size = 5
            y_min, y_max = max(0, pixel_y - box_size), min(self.latest_depth_img.shape[0], pixel_y + box_size)
            x_min, x_max = max(0, pixel_x - box_size), min(self.latest_depth_img.shape[1], pixel_x + box_size)
            
            depth_roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
            valid_depths = depth_roi[depth_roi > 0]

            if len(valid_depths) > 0:
                depth_mm = np.median(valid_depths)
                z_raw = depth_mm / 1000.0
                x_raw = (pixel_x - self.cx) * z_raw / self.fx
                y_raw = (pixel_y - self.cy) * z_raw / self.fy

                # --- DATA COLLECTION ENGINE ---
                if self.is_collecting:
                    self.measurement_buffer.append((x_raw, y_raw, z_raw))
                    
                    # Update video text to show progress
                    cv2.putText(frame, f"COLLECTING: {len(self.measurement_buffer)}/{self.samples_needed}", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Check if we have enough samples!
                    if len(self.measurement_buffer) >= self.samples_needed:
                        self.process_and_publish_average()

                else:
                    cv2.putText(frame, "WAITING FOR TRIGGER", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not target_found:
            status_text = "SEARCHING..." if self.is_collecting else "WAITING FOR TARGET..."
            cv2.putText(frame, status_text, (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        # Always publish video so RViz doesn't freeze
        overlay_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.overlay_pub.publish(overlay_msg)

    def process_and_publish_average(self):
        # Stop collecting
        self.is_collecting = False
        
        # Calculate the Mean
        avg_x = sum(p[0] for p in self.measurement_buffer) / self.samples_needed
        avg_y = sum(p[1] for p in self.measurement_buffer) / self.samples_needed
        avg_z = sum(p[2] for p in self.measurement_buffer) / self.samples_needed

        self.get_logger().info(f"Done! Averaged Target at: X={avg_x:.3f}, Y={avg_y:.3f}, Z={avg_z:.3f}")

        # 1. Publish the final 3D Point
        target_msg = PointStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = "oak_rgb_camera_optical_frame" 
        target_msg.point.x = float(avg_x)
        target_msg.point.y = float(avg_y)
        target_msg.point.z = float(avg_z)
        self.target_pub.publish(target_msg)

        # 2. Publish the RViz Marker
        marker = Marker()
        marker.header.stamp = target_msg.header.stamp
        marker.header.frame_id = "oak_rgb_camera_optical_frame"
        marker.ns = "weeds"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(avg_x)
        marker.pose.position.y = float(avg_y)
        marker.pose.position.z = float(avg_z)
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.0
        marker.color.a = 1.0 # Solid alpha
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = TriggeredOrangeTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()