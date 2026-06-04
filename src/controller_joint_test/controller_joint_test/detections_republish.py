import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range 
from std_msgs.msg import Int32

class DetectionsRepublishNode(Node):
    def __init__(self):
        super().__init__('plant_republish_node')
        
        # 1. State Variables
        self.latest_tof_z = None
        self.is_measuring = False
        self.target_samples = 0
        self.measurement_buffer = []
        
        self.locked_class_id = None 
        self.score_limit = 0.8
        
        # Subscribers
        self.detection_sub = self.create_subscription(Detection3DArray, '/oak/nn/spatial_detections', self.detection_callback, 10)
        self.tof_sub = self.create_subscription(Range, '/range', self.tof_callback, 10)
        self.trigger_sub = self.create_subscription(Int32, '/trigger_measurement', self.trigger_callback, 10)

        # Publisher
        self.detection_republish = self.create_publisher(Point, '/weed_location_cam_frame', 10)

        self.get_logger().info('Plant Measurement node ready! Waiting for trigger on /trigger_measurement...')

    def trigger_callback(self, msg):
        if msg.data <= 0:
            self.get_logger().warn(f'Invalid sample size requested: {msg.data}. Must be > 0.')
            return

        self.target_samples = msg.data
        self.measurement_buffer.clear() 
        self.locked_class_id = None     
        self.is_measuring = True       
        
        self.get_logger().info(f'Trigger received! Collecting {self.target_samples} samples...')

    def tof_callback(self, msg):
        self.latest_tof_z = msg.range

    def detection_callback(self, msg):
        if not self.is_measuring:
            return 
            
        if len(msg.detections) == 0:
            return 

        if self.latest_tof_z is None:
            self.get_logger().warn('Weed detected, but waiting for ToF sensor data...', throttle_duration_sec=2.0)
            return

        # Find the weed we want to measure
        target_det = None

        if self.locked_class_id is None:
            # We haven't locked onto a weed yet. Find the first valid one in this frame.
            for det in msg.detections:
                if det.results[0].hypothesis.score > self.score_limit:
                    target_det = det
                    self.locked_class_id = det.results[0].hypothesis.class_id
                    self.get_logger().info(f'Locked onto weed class: {self.locked_class_id}')
                    break
        else:
            # We are already locked on. Search this frame for our specific weed class.
            for det in msg.detections:
                if det.results[0].hypothesis.class_id == self.locked_class_id and det.results[0].hypothesis.score > self.score_limit:
                    target_det = det
                    break # Found our target, stop looking

        # If our locked target isn't in this specific frame (or score is too low), skip this frame
        if target_det is None:
            self.get_logger().warn(f'Looking for class {self.locked_class_id}, but could not find a confident match in this frame.', throttle_duration_sec=1.0)
            return

        # --- Valid Sample Found! ---
        # Keep X and Y from the Oak-D Camera
        x_val = target_det.results[0].pose.pose.position.x
        y_val = target_det.results[0].pose.pose.position.y
        # Inject the Z from the ToF sensor
        z_val = float(self.latest_tof_z)

        self.measurement_buffer.append((x_val, y_val, z_val))
        
        current_count = len(self.measurement_buffer)
        self.get_logger().info(f'Sample {current_count}/{self.target_samples} collected.')

        # Check if we have hit our requested target size
        if current_count >= self.target_samples:
            
            # Calculate the final average
            avg_x = sum(p[0] for p in self.measurement_buffer) / self.target_samples
            avg_y = sum(p[1] for p in self.measurement_buffer) / self.target_samples
            avg_z = sum(p[2] for p in self.measurement_buffer) / self.target_samples

            # Publish the smoothed result
            final_target = Point()
            final_target.x = avg_x
            final_target.y = avg_y
            final_target.z = avg_z
            self.detection_republish.publish(final_target)

            self.get_logger().info(f'✅ Success! Published avg target [Class {self.locked_class_id}]: X:{avg_x:.3f}, Y:{avg_y:.3f}, Z:{avg_z:.3f}')
            
            # Reset the state machine back to idle
            self.is_measuring = False
            self.measurement_buffer.clear()
            self.locked_class_id = None

def main(args=None):
    rclpy.init(args=args)
    node = DetectionsRepublishNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()