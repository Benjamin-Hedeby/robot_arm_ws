#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
from vision_msgs.msg import Detection3DArray
import time
import numpy as np
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class ObjectDetectorPublisher(Node):
    def __init__(self):
        super().__init__('object_detector_node')
        self.bridge = CvBridge()
        
        # Define QoS profile before creating publishers
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(String, 'object_detection/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'object_detection/markers', sensor_qos)
        self.image_pub = self.create_publisher(Image, 'object_detection/image_raw', sensor_qos)
        self.depth_pub = self.create_publisher(Image, 'object_detection/depth_raw', 10)
        
        # Keep track of the last time the synced callback ran
        self.last_sync_time = time.time()
        self.latest_depth_image = None

        # Parameters
        self.declare_parameter("detection_sync_slop", 0.1)
        self.sync_slop = self.get_parameter("detection_sync_slop").value
        self.get_logger().info(f"Using detection sync slop: {self.sync_slop} seconds")

        self.declare_parameter("bbox_scale_factor", 0.8)
        self.bbox_scale_factor = self.get_parameter("bbox_scale_factor").value
        self.get_logger().info(f"Using bounding box scale factor: {self.bbox_scale_factor}")
        
        self.declare_parameter("center_proximity_threshold", 100)
        self.center_proximity_threshold = self.get_parameter("center_proximity_threshold").value
        self.get_logger().info(f"Using center proximity threshold: {self.center_proximity_threshold} pixels")

        self.declare_parameter("image_scale_factor", 0.5)
        self.scale_factor = self.get_parameter("image_scale_factor").value
        self.get_logger().info(f"Using image scale factor: {self.scale_factor}")

        # Subscribe to image and detection topics
        self.image_sub = message_filters.Subscriber(self, Image, '/oak/rgb/image_rect')
        self.detection_sub = message_filters.Subscriber(self, Detection3DArray, '/oak/nn/spatial_detections')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.detection_sub],
            queue_size=10,
            slop=self.sync_slop)
        self.ts.registerCallback(self.synced_callback)

        # Subscribe to depth image
        self.depth_sub = self.create_subscription(
            Image,
            '/oak/stereo/image_raw',
            self.depth_callback,
            10
        )
        
        self.get_logger().info("Object Detector node started.")
        
        # Add as class member in __init__
        self._last_frame_time = self.get_clock().now()

    def depth_callback(self, depth_msg):
        """Process the depth image and publish it"""
        try:
            # Convert to OpenCV format
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg)
            self.latest_depth_image = cv_depth.copy()
            
            # Republish the depth image for visualization
            self.depth_pub.publish(depth_msg)
            
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion error: {e}")

    def synced_callback(self, image_msg, detection_msg):
        # Add at the beginning of image callback after getting the message
        current_time = self.get_clock().now()
        if (current_time - self._last_frame_time).nanoseconds / 1e9 < 0.05:  # 20 fps max
            return
        self._last_frame_time = current_time

        # Update our sync timestamp
        self.last_sync_time = time.time()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            # Add right after converting image with bridge.imgmsg_to_cv2
            if self.scale_factor != 1.0:
                h, w = cv_image.shape[:2]
                new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
                cv_image = cv2.resize(cv_image, (new_w, new_h))
            
            # Create a copy for annotations
            annotated_image = cv_image.copy()
            
            image_h, image_w, _ = cv_image.shape
            
            # Define image center coordinates here, outside the detection loop
            # so they're always available
            image_center_x = image_w // 2
            image_center_y = image_h // 2

            # Dictionary to count detections per label
            label_count = {}

            if detection_msg.detections:
                for detection in detection_msg.detections:
                    if not detection.results:
                        continue
                    result = detection.results[0]

                    try:
                        # Extract spatial coordinates from the result's pose
                        x_coord = result.pose.pose.position.x
                        y_coord = result.pose.pose.position.y
                        z_coord = result.pose.pose.position.z
                        
                        # Extract bounding box directly using the bbox center and size
                        center = detection.bbox.center
                        size = detection.bbox.size

                        # Network size is confirmed to be 300x300 from the debug logs
                        network_width = 300
                        network_height = 300

                        # Get center point of detection
                        cx = center.position.x / network_width
                        cy = center.position.y / network_height

                        # Get original width and height
                        original_width = size.x / network_width
                        original_height = size.y / network_height

                        # Apply the scaling factor to the width and height
                        scaled_width = original_width * self.bbox_scale_factor
                        scaled_height = original_height * self.bbox_scale_factor

                        # Calculate corners using the scaled dimensions
                        x_min = cx - scaled_width/2
                        y_min = cy - scaled_height/2
                        x_max = cx + scaled_width/2
                        y_max = cy + scaled_width/2

                        # Clip values to ensure they're between 0-1
                        x_min = max(0.0, min(1.0, x_min))
                        y_min = max(0.0, min(1.0, y_min))
                        x_max = max(0.0, min(1.0, x_max))
                        y_max = max(0.0, min(1.0, y_max))

                        # Scale to actual image dimensions
                        x1 = int(x_min * image_w)
                        y1 = int(y_min * image_h)
                        x2 = int(x_max * image_w)
                        y2 = int(y_max * image_h)

                        # Draw the bounding box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Calculate centers - no need to redefine image_center_x and image_center_y here
                        # since they're already defined above
                        bbox_center_x = (x1 + x2) // 2
                        bbox_center_y = (y1 + y2) // 2

                        # Draw a line from bbox center to image center
                        cv2.line(annotated_image, (bbox_center_x, bbox_center_y), 
                                 (image_center_x, image_center_y), (0, 0, 255), 2)

                        # Calculate displacement vector, angle and distance
                        dx = bbox_center_x - image_center_x
                        dy = bbox_center_y - image_center_y
                        distance = np.sqrt(dx*dx + dy*dy)
                        angle = np.degrees(np.arctan2(dy, dx))

                        # Define label first
                        if result.hypothesis.class_id:
                            label = result.hypothesis.class_id
                        else:
                            label = "Object"

                        # Use the top-left corner for label placement
                        label_x, label_y = x1, y1 - 10
                            
                        # Format coordinates with sign preserved, distance in mm
                        x_mm = int(x_coord * 1000)
                        y_mm = int(y_coord * 1000)
                        z_mm = int(z_coord * 1000)

                        # Get depth value at the center of the image from depth map
                        center_depth_mm = 0  # Default value if depth is not available
                        if self.latest_depth_image is not None:
                            try:
                                # Get depth value from center of the image
                                center_depth = self.latest_depth_image[image_center_y, image_center_x]
                                if center_depth > 0:  # Ensure valid depth value
                                    center_depth_mm = int(center_depth)
                            except IndexError:
                                # Handle potential index errors if depth image size doesn't match
                                pass

                        # Now publish complete detection information
                        detection_string = String()
                        detection_string.data = (f"{label},{angle:.1f},{distance:.1f},{x_mm},{y_mm},{z_mm},{center_depth_mm}")
                        self.detection_pub.publish(detection_string)

                        # Create a marker for visualization
                        marker_array = MarkerArray()
                        marker = Marker()
                        marker.header.frame_id = "oak_rgb_camera_optical_frame"
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.ns = "detections"
                        marker.id = 0
                        marker.type = Marker.ARROW
                        marker.action = Marker.ADD
                        marker.pose.position.x = x_coord
                        marker.pose.position.y = y_coord
                        marker.pose.position.z = z_coord
                        marker.pose.orientation.w = 1.0  # Default orientation (identity quaternion)
                        marker.scale.x = 0.1  # Arrow length
                        marker.scale.y = 0.01  # Arrow width
                        marker.scale.z = 0.01  # Arrow height
                        marker.color.r = 1.0   # Red color
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0   # Full opacity
                        marker.lifetime.sec = 0  # Persist until next update
                        marker_array.markers.append(marker)
                        self.marker_pub.publish(marker_array)

                        # Check if object is close to center (red line is shorter than threshold)
                        if distance < self.center_proximity_threshold:
                            # Get depth value at the center of the image from depth map
                            if self.latest_depth_image is not None:
                                # Get depth value from center of the image
                                center_depth = self.latest_depth_image[image_center_y, image_center_x]
                                
                                # Convert to meaningful distance (in mm)
                                if center_depth > 0:  # Ensure valid depth value
                                    # Display distance at the top of the screen in large text
                                    center_dist_mm = int(center_depth)
                                    cv2.putText(annotated_image, 
                                               f"DISTANCE TO OBJECT: {center_dist_mm} mm", 
                                               (image_w//4, 60), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                    
                                    # Draw a circular highlight around the target center area
                                    radius = int(self.center_proximity_threshold / 2)  # Circle diameter = threshold
                                    cv2.circle(annotated_image, 
                                              (image_center_x, image_center_y),
                                              radius,
                                              (0, 255, 255), 2)  # Yellow circle, 2px thickness

                        # Mark the centers for clarity
                        cv2.circle(annotated_image, (image_center_x, image_center_y), 5, (255, 255, 255), -1)
                        cv2.circle(annotated_image, (bbox_center_x, bbox_center_y), 5, (0, 255, 255), -1)

                        # Display the angle and distance
                        cv2.putText(annotated_image, 
                                    f"Angle: {angle:.1f}°, Dist: {distance:.1f}px", 
                                    (bbox_center_x + 10, bbox_center_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # Add overlay text
                        cv2.putText(annotated_image, label, (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Add spatial coordinates (convert to mm like in Luxonis example)
                        if x_coord != 0 or y_coord != 0 or z_coord != 0:
                            # The OAK's coordinate system: X+ right, Y+ down, Z+ forward
                            cv2.putText(annotated_image, f"X: {x_mm} mm", (label_x, label_y + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(annotated_image, f"Y: {y_mm} mm", (label_x, label_y + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(annotated_image, f"Z: {z_mm} mm", (label_x, label_y + 45),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Add to label_count
                        if label in label_count:
                            label_count[label] += 1
                        else:
                            label_count[label] = 1

                    except AttributeError as e:
                        self.get_logger().warn(f"Could not process detection bbox: {e}")
                        continue

            # Create summary overlay
            if label_count:
                summary_parts = []
                for lbl, count in label_count.items():
                    summary_parts.append(f"{count} objects of type {lbl} detected")
                summary_text = ", ".join(summary_parts)
            else:
                summary_text = "NO OBJECTS DETECTED"

            # Add FPS counter
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_sync_time + 0.001)
            cv2.putText(annotated_image, f"FPS: {fps:.2f}", (2, image_h - 4), 
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255), 1)

            # ALWAYS display the depth at the center of the camera view with greater prominence
            if self.latest_depth_image is not None:
                # Mark the center point with a crosshair
                crosshair_size = 10
                cv2.line(annotated_image, 
                         (image_center_x - crosshair_size, image_center_y),
                         (image_center_x + crosshair_size, image_center_y),
                         (0, 255, 255), 1)
                cv2.line(annotated_image, 
                         (image_center_x, image_center_y - crosshair_size),
                         (image_center_x, image_center_y + crosshair_size),
                         (0, 255, 255), 1)
                
                # Get depth value at center and display it
                try:
                    center_depth = self.latest_depth_image[image_center_y, image_center_x]
                    if center_depth > 0:  # Valid depth
                        center_depth_mm = int(center_depth)
                        
                        # Draw a prominent display of the center depth
                        # Background rectangle for better visibility
                #        text = f"CENTER DEPTH: {center_depth_mm} mm"
                #        font = cv2.FONT_HERSHEY_SIMPLEX
                #        font_size = 0.7
                #        thickness = 2
                #        text_size, _ = cv2.getTextSize(text, font, font_size, thickness)
                #        
                #        # Position at top-right corner
                #        text_x = image_w - text_size[0] - 10
                #        text_y = 30
                #        
                #        # Draw semi-transparent background
                #        cv2.rectangle(annotated_image, 
                #                     (text_x - 5, text_y - text_size[1] - 5),
                #                     (text_x + text_size[0] + 5, text_y + 5),
                #                     (0, 0, 0), -1)
                #        
                #        # Draw text in bright yellow
                #        cv2.putText(annotated_image, text, (text_x, text_y),
                #                   font, font_size, (0, 255, 255), thickness)
                #        
                #        # Small dot at the center where depth is measured
                #        cv2.circle(annotated_image, 
                #                  (image_center_x, image_center_y),
                #                  3, (0, 255, 255), -1)
                #        
                #        # Also still keep the bottom text for consistency
                #        cv2.putText(annotated_image, 
                #                   f"Center depth: {center_depth_mm} mm", 
                #                   (10, image_h - 25), 
                #                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except IndexError:
                    # Handle potential index errors if depth image size doesn't match
                    pass

            cv2.putText(annotated_image, summary_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label_count else (0, 0, 255), 2)

            # Before publishing, resize back to original resolution (add right before cv2_to_imgmsg)
            if self.scale_factor != 1.0:
                annotated_image = cv2.resize(annotated_image, (w, h))

            # Publish the annotated image for the subscriber node
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
                annotated_msg.header = image_msg.header  # Preserve the original timestamp
                self.image_pub.publish(annotated_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"Error converting annotated image: {e}")

        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
