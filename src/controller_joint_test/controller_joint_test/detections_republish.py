import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Range 

class DetectionsRepublishNode(Node):
    def __init__(self):
        super().__init__('arm_node')
        
        # 1. Create a variable to remember the latest ToF reading
        self.latest_tof_z = None
        
        # Subscribers
        self.detection_sub = self.create_subscription(Detection3DArray, '/oak/nn/spatial_detections', self.detection_callback, 10)
        self.tof_sub = self.create_subscription(Range, '/range', self.tof_callback, 10)

        # Publisher
        self.detection_republish = self.create_publisher(Point, '/weed_location_cam_frame', 10)

    # 2. Implement the ToF Callback
    def tof_callback(self, msg):
        # Constantly update our memory with the newest laser distance
        self.latest_tof_z = msg.range

    def detection_callback(self, msg):
        if len(msg.detections) == 0:
            return # If no detections, do nothing and wait for the next frame
            
        # 3. Safety Check: Make sure the ToF sensor has actually sent data before we try to use it!
        if self.latest_tof_z is None:
            self.get_logger().warn('Weed detected, but waiting for ToF sensor data...', throttle_duration_sec=2.0)
            return
        
        score_limit = 0.8
        score = msg.detections[0].results[0].hypothesis.score

        if score > score_limit:
           detection = Point()

           # Keep X and Y from the Oak-D Camera
           detection.x = msg.detections[0].results[0].pose.pose.position.x
           detection.y = msg.detections[0].results[0].pose.pose.position.y
           
           # 4. Inject the Z from the ToF sensor!
           detection.z = float(self.latest_tof_z)

           self.detection_republish.publish(detection)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionsRepublishNode()
    rclpy.spin(node) # Keeps the node alive
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
