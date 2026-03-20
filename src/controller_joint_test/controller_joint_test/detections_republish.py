import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import Point

class DetectionsRepublishNode(Node):
    def __init__(self):
        super().__init__('arm_node')
        
        # Subscriber: Camera detections
        self.detection_sub = self.create_subscription(Detection3DArray, '/oak/nn/spatial_detections', self.detection_callback, 10)

        # Publisher: Republish x, y, z if score is high enough
        self.detection_republish = self.create_publisher(Point, '/weed_location_cam_frame', 10)

    def detection_callback(self, msg):

        if len(msg.detections) == 0:
            return # If no detections, do nothing and wait for the next frame
        
        score_limit = 0.8

        score = msg.detections[0].results[0].hypothesis.score

        if score > score_limit:

           detection = Point()

           detection.x = msg.detections[0].results[0].pose.pose.position.x
           detection.y = msg.detections[0].results[0].pose.pose.position.y
           detection.z = msg.detections[0].results[0].pose.pose.position.z

           self.detection_republish.publish(detection)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionsRepublishNode()
    rclpy.spin(node) # Keeps the node alive
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()