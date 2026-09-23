#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, JointState
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs_py.point_cloud2 as pc2
import cv2

from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs

from VisionTransform import transform_camera_to_base

class SpatialOverlay(Node):
    def __init__(self):
        super().__init__('spatial_overlay')
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None

        # TF2 Setup (Only used for drawing back onto the 2D image)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.fixed_frame = 'link_0' 
        
        self.pending_point = None
        self.world_point = None
        self.current_joints = []

        # Publishers
        self.img_pub  = self.create_publisher(Image,       '/spatial_overlay/image_rect', 1)
        self.mark_pub = self.create_publisher(MarkerArray, '/spatial_overlay/markers',    1)
        self.pc_pub   = self.create_publisher(PointCloud2, '/spatial_overlay/points',     1)

        # Subscriptions
        self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.caminfo_cb, 10)
        self.create_subscription(Image, '/oak/rgb/image_rect', self.image_cb, 1)
        self.create_subscription(Point, '/weed_location_cam_frame', self.point_cb, 10)
        
        # New subscription to get robot joint angles
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

    def joint_cb(self, msg: JointState):
        self.current_joints = msg.position

    def caminfo_cb(self, msg: CameraInfo):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def point_cb(self, msg: Point):
        # Store the new point until the next image frame processes it
        if self.world_point is None:
            self.pending_point = msg

    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return

        overlay = frame.copy()
        ma = MarkerArray()

        # 1. Transform using your custom Forward Kinematics script
        if self.pending_point is not None and len(self.current_joints) >= 6:
            x_b, y_b, z_b = transform_camera_to_base(
                self.pending_point.x,
                self.pending_point.y,
                self.pending_point.z,
                self.current_joints
            )
            
            # Save as a standard Point object
            self.world_point = Point(x=float(x_b), y=float(y_b), z=float(z_b))
            self.pending_point = None
            self.get_logger().info('Position locked in base frame via VisionTransform.py')

        if self.world_point is None:
            # Delete markers if no point has been anchored yet
            clear_marker = Marker()
            clear_marker.header = msg.header
            clear_marker.ns = 'plants'
            clear_marker.action = Marker.DELETEALL
            ma.markers.append(clear_marker)
            self.mark_pub.publish(ma)

            cv2.putText(overlay, '0 detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 2)
        else:
            x_w = self.world_point.x
            y_w = self.world_point.y
            z_w = self.world_point.z

            # 2. RViz Marker in the fixed frame
            m = Marker()
            m.header.frame_id = self.fixed_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'plants'
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x_w
            m.pose.position.y = y_w
            m.pose.position.z = z_w
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0; m.color.a = 0.8
            ma.markers.append(m)
            self.mark_pub.publish(ma)

            # 3. PointCloud2 in the fixed frame
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            pc2_msg = pc2.create_cloud(
                header=m.header,
                fields=fields,
                points=[[x_w, y_w, z_w]]
            )
            self.pc_pub.publish(pc2_msg)

            cv2.putText(overlay, '1 weed tracked', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 2)

            # 4. Transform back to the current camera frame to draw the 2D overlay
            try:
                pt = PointStamped()
                pt.header.frame_id = self.fixed_frame
                pt.point = self.world_point
                
                t_back = self.tf_buffer.lookup_transform(
                    msg.header.frame_id,
                    self.fixed_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                curr_cam_pt = tf2_geometry_msgs.do_transform_point(pt, t_back)
                x_c = curr_cam_pt.point.x
                y_c = curr_cam_pt.point.y
                z_c = curr_cam_pt.point.z

                if z_c > 0.0 and None not in (self.fx, self.fy, self.cx, self.cy):
                    u = int(self.fx * x_c / z_c + self.cx)
                    v = int(self.fy * y_c / z_c + self.cy)
                    cv2.circle(overlay, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(overlay, f'{z_c:.2f}m', (u + 6, v - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except TransformException:
                pass

        try:
            out = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
            out.header = msg.header
            self.img_pub.publish(out)
        except CvBridgeError:
            pass

def main():
    rclpy.init()
    node = SpatialOverlay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()