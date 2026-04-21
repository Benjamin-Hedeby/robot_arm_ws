import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Range 
from collections import deque  # Perfect for creating a rolling memory buffer

class DetectionsRepublishNode(Node):
    def __init__(self):
        super().__init__('arm_node')
        
        # 1. Create a variable to remember the latest ToF reading
        self.latest_tof_z = None
        
        # --- SAFETY LIMITS ---
        self.max_abs_xy = 0.5 
        
        # --- AVERAGING SETTINGS ---
        self.num_samples = 50
        # deque automatically deletes the oldest item when it hits the maxlen!
        self.measurement_buffer = deque(maxlen=self.num_samples)
        
        # Subscribers
        self.detection_sub = self.create_subscription(PointStamped, '/orange_target_3d', self.detection_callback, 10)
        self.tof_sub = self.create_subscription(Range, '/range', self.tof_callback, 10)

        # Publisher
        self.detection_republish = self.create_publisher(Point, '/weed_location_cam_frame', 10)

    def tof_callback(self, msg):
        self.latest_tof_z = msg.range

    def detection_callback(self, msg):
        # 1. Safety Check: Make sure the ToF sensor has actually sent data
        if self.latest_tof_z is None:
            self.get_logger().warn('Orange target detected, but waiting for ToF sensor data...', throttle_duration_sec=2.0)
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

        # 4. Check if we have enough samples to make an average
        if len(self.measurement_buffer) < self.num_samples:
            self.get_logger().info(
                f'Collecting samples to stabilize... ({len(self.measurement_buffer)}/{self.num_samples})', 
                throttle_duration_sec=0.5)
            return

        # 5. Calculate the average of all 20 stored measurements
        avg_x = sum(p[0] for p in self.measurement_buffer) / self.num_samples
        avg_y = sum(p[1] for p in self.measurement_buffer) / self.num_samples
        avg_z = sum(p[2] for p in self.measurement_buffer) / self.num_samples

        # 6. Publish the smoothed result!
        detection = Point()
        detection.x = avg_x
        detection.y = avg_y
        detection.z = avg_z

        self.detection_republish.publish(detection)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionsRepublishNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()