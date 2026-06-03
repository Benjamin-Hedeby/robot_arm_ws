import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range 
from std_msgs.msg import Int32

class DetectionsRepublishNode(Node):
    def __init__(self):
        super().__init__('plant_republish_node')
        
        # --- CONFIGURATION ---
        self.score_limit = 0.5
        
        # --- STATE VARIABLES ---
        self.latest_tof_z = None
        self.state = 'IDLE'  # Can be 'IDLE', 'SEARCHING', or 'MEASURING'
        self.target_samples = 0
        self.measurement_buffer = []
        self.locked_class_id = None 
        
        # Subscribers
        self.detection_sub = self.create_subscription(Detection3DArray, '/oak/nn/spatial_detections', self.detection_callback, 10)
        self.tof_sub = self.create_subscription(Range, '/range', self.tof_callback, 10)
        self.trigger_sub = self.create_subscription(Int32, '/trigger_measurement', self.trigger_callback, 10)

        # Publisher
        self.detection_republish = self.create_publisher(Point, '/weed_location_cam_frame', 10)

        self.get_logger().info('Plant Measurement node ready! Waiting for Task Controller trigger...')

    def trigger_callback(self, msg):
        if msg.data <= 0:
            self.state = 'IDLE'
            self.get_logger().info('Trigger <= 0 received. Node is now IDLE.')
            return

        # Setup the new measurement task
        self.target_samples = msg.data
        self.measurement_buffer.clear() 
        self.locked_class_id = None
        
        # Wake up and wait for the first weed
        self.state = 'SEARCHING' 
        self.get_logger().info(f'Trigger received! Waiting for a weed to enter the frame...')

    def tof_callback(self, msg):
        self.latest_tof_z = msg.range

    def detection_callback(self, msg):
        # 1. Ignore camera entirely if we are asleep (IDLE)
        if self.state == 'IDLE':
            return 
            
        # 2. Handle empty frames gracefully
        if len(msg.detections) == 0:
            if self.state == 'SEARCHING':
                self.get_logger().info('Waiting for weed...', throttle_duration_sec=2.0)
            return 

        # 3. Safety Check for ToF Sensor
        if self.latest_tof_z is None:
            self.get_logger().warn('Weed detected, but waiting for ToF sensor data...', throttle_duration_sec=2.0)
            return

        target_det = None

        # 4. State: SEARCHING (Looking for the first weed to lock onto)
        if self.state == 'SEARCHING':
            for det in msg.detections:
                if det.results[0].hypothesis.score > self.score_limit:
                    target_det = det
                    self.locked_class_id = det.results[0].hypothesis.class_id
                    self.state = 'MEASURING'
                    self.get_logger().info(f'Target Acquired! Locked onto weed class: {self.locked_class_id}')
                    break
                    
        # 5. State: MEASURING (Extracting data for our locked weed)
        elif self.state == 'MEASURING':
            for det in msg.detections:
                # Ensure it's the exact same type of weed, and high confidence
                if det.results[0].hypothesis.class_id == self.locked_class_id and det.results[0].hypothesis.score > self.score_limit:
                    target_det = det
                    break 

        # 6. If our locked target vanished from this specific frame
        if target_det is None:
            if self.state == 'MEASURING':
                self.get_logger().warn(f'Lost track of class {self.locked_class_id}! Waiting for it to reappear...', throttle_duration_sec=1.0)
            return

        # --- VALID SAMPLE FOUND! ---
        x_val = target_det.results[0].pose.pose.position.x
        y_val = target_det.results[0].pose.pose.position.y
        z_val = float(self.latest_tof_z)

        self.measurement_buffer.append((x_val, y_val, z_val))
        
        current_count = len(self.measurement_buffer)
        self.get_logger().info(f'Sample {current_count}/{self.target_samples} collected.')

        # 7. Check if we have hit our requested target size
        if current_count >= self.target_samples:
            
            # Calculate the final average
            avg_x = sum(p[0] for p in self.measurement_buffer) / self.target_samples
            avg_y = sum(p[1] for p in self.measurement_buffer) / self.target_samples
            avg_z = sum(p[2] for p in self.measurement_buffer) / self.target_samples

            # Publish the smoothed result back to the Task Controller
            final_target = Point()
            final_target.x = avg_x
            final_target.y = avg_y
            final_target.z = avg_z
            self.detection_republish.publish(final_target)

            self.get_logger().info(f'✅ Success! Published avg target [Class {self.locked_class_id}]: X:{avg_x:.3f}, Y:{avg_y:.3f}, Z:{avg_z:.3f}')
            
            # 8. Mission Accomplished: Go back to sleep until the next trigger
            self.measurement_buffer.clear()
            self.locked_class_id = None
            self.state = 'IDLE'
            self.get_logger().info('Measurement complete. Node is IDLE. Awaiting next trigger.')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionsRepublishNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()