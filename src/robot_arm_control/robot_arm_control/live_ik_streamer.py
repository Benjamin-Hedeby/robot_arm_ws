import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from .configuration import JOINT_LIMITS

# Import your validated IK function
from .InverseKinematics import inverse_kinematics

class LiveIKStreamer(Node):
    def __init__(self):
        super().__init__('live_ik_streamer')
        
        # 1. Subscriber: Listens for an array of 6 floats [X, Y, Z, R, P, Y]
        self.target_sub = self.create_subscription(
            Float64MultiArray,
            '/desired_tcp_pose_euler',
            self.pose_callback,
            10
        )
        
        # 2. Publisher: Sends the calculated joint angles
        self.joint_pub = self.create_publisher(
            Float64MultiArray,
            '/arm_controller/commands',
            10
        )
        
        self.get_logger().info("Live IK Streamer running! Expecting [X, Y, Z, Roll, Pitch, Yaw]")

    def pose_callback(self, msg):
        try:
            # Safety check: make sure the terminal command actually sent 6 numbers
            if len(msg.data) != 6:
                self.get_logger().warn(f"Expected 6 values [X,Y,Z,R,P,Y], but got {len(msg.data)}")
                return

            # Extract position (Meters)
            position = [msg.data[0], msg.data[1], msg.data[2]]
            
            # Extract orientation (Euler angles in Radians)
            orientation = [msg.data[3], msg.data[4], msg.data[5]]
            
            # Run the Inverse Kinematics calculation
            joints = inverse_kinematics(position, orientation)
            
            # Check the first 5 physical joints against their limits
            limit_exceeded = False
            for i in range(5):
                if not (JOINT_LIMITS[i][0] <= joints[i] <= JOINT_LIMITS[i][1]):
                    self.get_logger().error(
                        f"SAFETY TRIGGERED! Joint {i+1} requested angle {round(joints[i], 3)} rad "
                        f"is out of physical bounds [{round(JOINT_LIMITS[i][0], 2)}, {round(JOINT_LIMITS[i][1], 2)}]."
                    )
                    limit_exceeded = True
            
            # Abort the entire movement if any joint is dangerous
            if limit_exceeded:
                self.get_logger().warn("Move aborted to prevent self-collision!")
                return 
            # -----------------------------------
            
            # Print the result to the terminal
            #self.get_logger().info(f"Calculated Joint Angles: {[round(j, 3) for j in joints]}", throttle_duration_sec=2.0)

            # Changed from 6DOF to 5DOF:
            physical_joints = joints[:5]
            #physical_joints[4] = physical_joints[4] + 0.05
            
            # Round to 4 decimal places to prevent scientific notation (e-18) from crashing the hardware controllers.
            cleaned_joints = [round(float(j), 4) for j in physical_joints]
            
            # If a number is just negative zero (-0.0), force it to absolute 0.0
            cleaned_joints = [0.0 if j == -0.0 else j for j in cleaned_joints]
            
            # Pack and publish
            cmd_msg = Float64MultiArray()
            cmd_msg.data = cleaned_joints 
            
            self.joint_pub.publish(cmd_msg)
            
            self.get_logger().info(f"Sent clean joints: {cleaned_joints}", throttle_duration_sec=2.0)

        except Exception as e:
            self.get_logger().error(f"Failed to stream IK: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LiveIKStreamer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()