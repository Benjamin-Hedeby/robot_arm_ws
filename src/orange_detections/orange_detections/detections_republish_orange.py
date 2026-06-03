import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Range 
from std_msgs.msg import Int32  # Added to receive the trigger command

class DetectionsRepublishNode(Node):
    def __init__(self):
        super().__init__('arm_node')
        
        # 1. State Variables
        self.latest_tof_z = None
        self.is_measuring = False
        self.target_samples = 0
        self.measurement_buffer = []
        
        # --- SAFETY LIMITS ---
        self.max_abs_xy = 0.3
        
        # Subscribers
        self.detection_sub = self.create_subscription(PointStamped, '/orange_target_3d', self.detection_callback, 10)
        self.tof_sub = self.create_subscription(Range, '/range', self.tof_callback, 10)
        
        # NEW: Listens for the command to start measuring
        self.trigger_sub = self.create_subscription(Int32, '/trigger_measurement', self.trigger_callback, 10)

        # Publisher
        self.detection_republish = self.create_publisher(Point, '/weed_location_cam_frame', 10)

        self.get_logger().info('Measurement node ready! Waiting for trigger on /trigger_measurement...')

    def trigger_callback(self, msg):
        """Activates when someone sends an integer to the trigger topic."""
        if msg.data <= 0:
            self.get_logger().warn(f'Invalid sample size requested: {msg.data}. Must be > 0.')
            return

        self.target_samples = msg.data
        self.measurement_buffer.clear() # Reset the memory
        self.is_measuring = True        # Open the gates!
        
        self.get_logger().info(f'Trigger received! Collecting {self.target_samples} samples...')

    def tof_callback(self, msg):
        self.latest_tof_z = msg.range

    def detection_callback(self, msg):
        # 0. State Check: If we aren't commanded to measure, ignore the camera completely!
        if not self.is_measuring:
            return

        # 1. Safety Check: Make sure the ToF sensor has actually sent data
        if self.latest_tof_z is None:
            self.get_logger().warn('Waiting for ToF sensor data...', throttle_duration_sec=2.0)
            return
            
        # 2. Safety Check: Filter out wild X/Y outliers
        if abs(msg.point.x) > self.max_abs_xy or abs(msg.point.y) > self.max_abs_xy:
            self.get_logger().warn(
                f'Ignored outlier! X: {msg.point.x:.2f}m, Y: {msg.point.y:.2f}m is outside limits.', 
                throttle_duration_sec=1.0)
            return
        
        # 3. Store the valid measurements in our buffer
        z_val = float(self.latest_tof_z) - 0.019
        self.measurement_buffer.append((msg.point.x, msg.point.y, z_val))

        # Print progress so you know it's working
        current_count = len(self.measurement_buffer)
        self.get_logger().info(f'Sample {current_count}/{self.target_samples} collected.')

        # 4. Check if we have hit our requested target size
        if current_count >= self.target_samples:
            
            # Calculate the final average
            avg_x = sum(p[0] for p in self.measurement_buffer) / self.target_samples
            avg_y = sum(p[1] for p in self.measurement_buffer) / self.target_samples
            avg_z = sum(p[2] for p in self.measurement_buffer) / self.target_samples

            # Publish the smoothed result
            detection = Point()
            detection.x = avg_x
            detection.y = avg_y
            detection.z = avg_z
            self.detection_republish.publish(detection)

            self.get_logger().info(f'Success! Published avg target: X:{avg_x:.3f}, Y:{avg_y:.3f}, Z:{avg_z:.3f}')
            
            # Reset the state machine back to idle
            self.is_measuring = False
            self.measurement_buffer.clear()

def main(args=None):
    rclpy.init(args=args)
    node = DetectionsRepublishNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()