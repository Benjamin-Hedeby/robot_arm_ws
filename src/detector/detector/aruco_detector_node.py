#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

def draw_axis(image, camera_matrix, dist_coeffs, rvec, tvec, length):
    """
    Draw a 3D coordinate axis on the image by projecting points with cv2.projectPoints.
    The x-axis is drawn in red, y-axis in green, and z-axis in blue.
    Z-axis is computed as the cross product (normal) of the x and y axes.
    """
    # Define x and y axes
    x_axis = np.array([length, 0, 0])
    y_axis = np.array([0, length, 0])
    
    # Calculate z-axis as the cross product of x and y vectors (normal to xy plane)
    z_axis = np.cross(x_axis, y_axis)
    # Normalize and scale to match the length of x and y axes
    z_axis = z_axis / np.linalg.norm(z_axis) * length
    
    # Define the 3D points for projection
    axis_points = np.float32([
        [0, 0, 0],       # Origin
        x_axis,          # X-axis endpoint
        y_axis,          # Y-axis endpoint
        z_axis           # Z-axis endpoint (computed as normal)
    ]).reshape(-1, 3)

    # Project the 3D points to 2D image points
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    origin = tuple(imgpts[0])
    
    # Draw the axes lines: x in red, y in green, z in blue
    cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # x-axis (red)
    cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # y-axis (green)
    cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # z-axis (blue)

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.bridge = CvBridge()
        
        # Define QoS profile before creating publishers
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create publishers for the annotated image and marker data
        self.image_pub = self.create_publisher(Image, 'aruco_detection/image_raw', sensor_qos)
        self.marker_pub = self.create_publisher(String, 'aruco_detection/markers', sensor_qos)
        
        # Subscribe to the rectified image topic
        self.image_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_rect',  # Update if your topic name is different.
            self.image_callback,
            10
        )
        
        # Use the 5x5 ArUco dictionary
        #self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        # Use the 4x4 ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # Optimize detector parameters
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # Skip refinement
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        
        # Camera intrinsic parameters (update these with actual calibration)
        self.camera_matrix = np.array([[600.0, 0, 320],
                                       [0, 600.0, 240],
                                       [0, 0, 1]])
        self.dist_coeffs = np.zeros((5,))  # Assuming minimal distortion
        
        # Set the physical size of your marker (in meters)
        self.marker_length = 0.1
        
        # Add in __init__ method after other parameters
        self.declare_parameter("image_scale_factor", 0.5)
        self.scale_factor = self.get_parameter("image_scale_factor").value
        self.get_logger().info(f"Using image scale factor: {self.scale_factor}")
        
        self.get_logger().info("Aruco detector publisher node started.")
        
        # Add as class member in __init__
        self._last_frame_time = self.get_clock().now()

    def image_callback(self, msg):
        # Add at the beginning of image callback after getting the message
        current_time = self.get_clock().now()
        if (current_time - self._last_frame_time).nanoseconds / 1e9 < 0.05:  # 20 fps max
            return
        self._last_frame_time = current_time
        
        try:
            # Convert the incoming ROS image message to a BGR OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # Add right after converting image with bridge.imgmsg_to_cv2
        if self.scale_factor != 1.0:
            h, w = cv_image.shape[:2]
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            cv_image = cv2.resize(cv_image, (new_w, new_h))
        
        # Create a copy for annotations
        annotated_image = cv_image.copy()
        
        # Get image dimensions
        image_h, image_w, _ = cv_image.shape
        
        # Define image center coordinates
        image_center_x = image_w // 2
        image_center_y = image_h // 2
        
        # Always draw a crosshair at the center of the image
        crosshair_size = 10
        cv2.line(annotated_image, 
                 (image_center_x - crosshair_size, image_center_y),
                 (image_center_x + crosshair_size, image_center_y),
                 (0, 255, 255), 1)
        cv2.line(annotated_image, 
                 (image_center_x, image_center_y - crosshair_size),
                 (image_center_x, image_center_y + crosshair_size),
                 (0, 255, 255), 1)
        
        # Convert image to grayscale for marker detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Detect ArUco markers in the grayscale image
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        # List to store marker data
        marker_data = []
        
        # For tracking the closest marker
        min_distance = float('inf')
        closest_marker_info = None  # Will store (marker_id, distance_mm) for closest marker
        
        # Global overlay with text indicating overall detection status
        if ids is None or len(ids) == 0:
            cv2.putText(annotated_image, "NO MARKERS DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            marker_count = len(ids)
            text_global = f"{marker_count} MARKER{'S' if marker_count > 1 else ''} DETECTED"
            cv2.putText(annotated_image, text_global, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Estimate pose for each detected marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            # Draw the detected marker boundaries
            cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)
            
            # Center proximity threshold (same as in spatial tracker)
            center_proximity_threshold = 100
            
            # Draw a circular highlight area around the center
            # We'll show this if ANY marker is within the threshold
            show_highlight = False
            
            for i, marker_corner in enumerate(corners):
                marker_id = int(ids[i][0])
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                
                # Calculate marker center point
                marker_center_x = 0
                marker_center_y = 0
                for corner in marker_corner[0]:
                    marker_center_x += corner[0]
                    marker_center_y += corner[1]
                marker_center_x = int(marker_center_x / 4)
                marker_center_y = int(marker_center_y / 4)
                
                # Draw axes
                if hasattr(cv2.aruco, 'drawAxis'):
                    cv2.aruco.drawAxis(annotated_image, self.camera_matrix, self.dist_coeffs,
                                        rvec, tvec, self.marker_length * 0.5)
                else:
                    draw_axis(annotated_image, self.camera_matrix, self.dist_coeffs,
                              rvec, tvec, self.marker_length * 0.5)
                
                # Calculate displacement vector, angle and distance
                dx = marker_center_x - image_center_x
                dy = marker_center_y - image_center_y
                distance_px = np.sqrt(dx*dx + dy*dy)
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Convert 3D distance to mm
                distance_mm = int(tvec[2] * 1000)
                
                # Draw a line from marker center to image center
                cv2.line(annotated_image, (marker_center_x, marker_center_y), 
                         (image_center_x, image_center_y), (0, 0, 255), 2)
                
                # Mark the centers for clarity
                cv2.circle(annotated_image, (image_center_x, image_center_y), 5, (255, 255, 255), -1)
                cv2.circle(annotated_image, (marker_center_x, marker_center_y), 5, (0, 255, 255), -1)
                
                # Display the angle and distance
                cv2.putText(annotated_image, 
                            f"Angle: {angle:.1f}°, Dist: {distance_px:.1f}px", 
                            (marker_center_x + 10, marker_center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Define pink color for marker-specific info
                pink_color = (255, 0, 255)  # BGR format
                
                # Check if marker is close to center (within threshold)
                if distance_px < center_proximity_threshold:
                    # Set flag to show the highlight circle
                    show_highlight = True
                    
                    # Update closest marker if this one is closer
                    if distance_px < min_distance:
                        min_distance = distance_px
                        closest_marker_info = (marker_id, distance_mm)
                
                # Overlay text with the marker ID and translation (pose) - now in pink
                text_marker = f"ID:{marker_id} [{tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f}] {distance_mm}mm"
                # Place this text above the top-left corner of the detected marker
                top_left = marker_corner[0][0]
                pos = (int(top_left[0]), int(top_left[1]) - 10)
                cv2.putText(annotated_image, text_marker, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, pink_color, 2)
                
                # Store marker data
                marker_data.append(f"{marker_id},{tvec[0]:.3f},{tvec[1]:.3f},{tvec[2]:.3f},{angle:.1f},{distance_px:.1f}")
            
            # After processing all markers, check if we should draw highlight and distance text
            if show_highlight:
                # Draw the circular highlight
                radius = int(center_proximity_threshold / 2)
                cv2.circle(annotated_image, 
                          (image_center_x, image_center_y),
                          radius,
                          (0, 255, 255), 2)  # Yellow circle, 2px thickness
                
                # Display distance only for the closest marker
                if closest_marker_info is not None:
                    closest_id, closest_distance = closest_marker_info
                    cv2.putText(annotated_image, 
                               f"DISTANCE TO MARKER {closest_id}: {closest_distance} mm", 
                               (image_w//4, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, pink_color, 2)
        
        # Before publishing, resize back to original resolution (add right before cv2_to_imgmsg)
        if self.scale_factor != 1.0:
            annotated_image = cv2.resize(annotated_image, (w, h))
        
        # Publish the annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            annotated_msg.header = msg.header  # Preserve the original timestamp
            self.image_pub.publish(annotated_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting annotated image: {e}")
        
        # Publish marker data
        if marker_data:
            marker_msg = String()
            marker_msg.data = ";".join(marker_data)
            self.marker_pub.publish(marker_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
