#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class ObjectDetectorVisualizer(Node):
    def __init__(self):
        super().__init__('object_detector_visualizer')
        self.bridge = CvBridge()
        
        # Create display windows
        cv2.namedWindow("Object Feed", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
        
        # Last update time for FPS calculation
        self.last_update_time = time.time()
        self.latest_depth_image = None
        self.latest_rgb_image = None
        self.latest_detection = None
        
        # Parameters
        self.declare_parameter("center_proximity_threshold", 100)
        self.center_proximity_threshold = self.get_parameter("center_proximity_threshold").value
        self.get_logger().info(f"Using center proximity threshold: {self.center_proximity_threshold} pixels")
        
        # Define matching QoS profile - MUST MATCH PUBLISHER
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Use separate subscribers instead of synchronizer since String messages don't have headers
        self.create_subscription(
            Image,
            'object_detection/image_raw',
            self.image_callback,
            qos_profile=sensor_qos
        )
        
        self.create_subscription(
            String,
            'object_detection/status',
            self.detection_callback,
            qos_profile=sensor_qos
        )
        
        # Depth subscription
        self.depth_sub = self.create_subscription(
            Image,
            'object_detection/depth_raw',
            self.depth_callback,
            qos_profile=sensor_qos
        )
        
        # Create a timer for visualization updates
        self.timer = self.create_timer(0.1, self.update_visualization)  # 10 FPS
        
        self.get_logger().info("Object Detector Visualizer node started.")
    
    def depth_callback(self, depth_msg):
        """Process the depth image and display it"""
        try:
            # Convert to OpenCV format
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg)
            self.latest_depth_image = cv_depth.copy()
            
            # Create a colorized version of the depth image
            if cv_depth is not None:
                # Downsample more aggressively
                depth_downscaled = cv_depth[::4, ::4]  # Every 4th pixel instead of every 2nd
                
                # Simpler depth normalization
                depth_colormap = (depth_downscaled * 0.03).astype(np.uint8)  # Simple scaling instead of percentiles
                depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_HOT)
                
                # Resize to half size for display
                h, w = depth_colormap.shape[:2]
                display_depth = cv2.resize(depth_colormap, (w//2, h//2))
                
                # Display the colorized depth map
                cv2.imshow("Depth", display_depth)
                cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion error: {e}")
    
    def image_callback(self, image_msg):
        """Store the latest RGB image"""
        try:
            # Convert the image to OpenCV format and store it
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"RGB image conversion error: {e}")
    
    def detection_callback(self, detection_msg):
        """Store the latest detection data"""
        self.latest_detection = detection_msg.data
    
    def update_visualization(self):
        """Update the visualization with the latest data"""
        # Check if we have a valid RGB image
        if self.latest_rgb_image is not None:
            # Resize to half size for display
            h, w = self.latest_rgb_image.shape[:2]
            display_image = cv2.resize(self.latest_rgb_image, (w//2, h//2))
            
            # Display the RGB image
            cv2.imshow("Object Feed", display_image)
            cv2.waitKey(1)
            
            # Log that we're displaying the image 
            # Debug info - uncomment if needed
            # self.get_logger().info("Displaying RGB image")
            
            # Check if we also have detection data
            if self.latest_detection is not None:
                try:
                    # Parse the detection data (uncomment for debugging)
                    # self.get_logger().info(f"Detection data: {self.latest_detection}")
                    pass
                except Exception as e:
                    self.get_logger().error(f"Error parsing detection data: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()