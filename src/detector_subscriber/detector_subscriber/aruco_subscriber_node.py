#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class ArucoVisualizerNode(Node):
    def __init__(self):
        super().__init__('aruco_visualizer_node')
        self.bridge = CvBridge()
        
        # Create display windows
        cv2.namedWindow("Aruco Detection", cv2.WINDOW_AUTOSIZE)
        
        # Last image and marker data
        self.latest_image = None
        self.latest_markers = None
        
        # Define matching QoS profile - MUST MATCH PUBLISHER
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to image and marker data with matching QoS
        self.create_subscription(
            Image,
            'aruco_detection/image_raw',
            self.image_callback,
            qos_profile=sensor_qos  # Use matching QoS profile
        )
        
        self.create_subscription(
            String,
            'aruco_detection/markers',
            self.marker_callback,
            qos_profile=sensor_qos  # Use matching QoS profile
        )
        
        # Timer for updating the visualization
        self.timer = self.create_timer(0.1, self.update_visualization)  # 10 FPS
        
        self.get_logger().info("Aruco visualizer node started.")
    
    def image_callback(self, msg):
        """Process the image with ArUco detections"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
    
    def marker_callback(self, msg):
        """Process the ArUco marker data"""
        self.latest_markers = msg.data
    
    def update_visualization(self):
        """Update the visualization with the latest data"""
        if self.latest_image is not None:
            # Resize to half size for display
            h, w = self.latest_image.shape[:2]
            display_image = cv2.resize(self.latest_image, (w//2, h//2))
            
            # Display the image
            cv2.imshow("Aruco Detection", display_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
