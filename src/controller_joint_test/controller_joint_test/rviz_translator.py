import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.action import ActionServer
from control_msgs.action import GripperCommand

class RVizTranslator(Node):
    def __init__(self):
        super().__init__('rviz_translator')
        
        # 1. Listen for Arm Positions (Publisher/Subscriber)
        self.arm_sub = self.create_subscription(
            Float64MultiArray, 
            '/arm_controller/commands', 
            self.arm_command_callback, 
            10
        )
        
        # 2. Listen for Gripper Positions (Action Server)
        self.gripper_server = ActionServer(
            self,
            GripperCommand,
            '/gripper_action_controller/gripper_cmd',
            self.gripper_execute_callback
        )
        
        # 3. Publish to RViz
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # Exact joint names from your URDF (5 arm + 1 gripper)
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper1']
        
        # Start perfectly at zero! (Index 0-4 are arm, Index 5 is gripper)
        self.current_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Publish continuously at 30Hz like real hardware
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info("Digital Twin active! Mocking arm and gripper...")

    def arm_command_callback(self, msg):
        # Update just the first 5 elements (the arm joints)
        for i in range(5):
            self.current_positions[i] = msg.data[i]

    def gripper_execute_callback(self, goal_handle):
        target_pos = goal_handle.request.command.position
        self.get_logger().info(f"Digital Gripper moving to: {target_pos}")
        
        # Update the 6th element (the gripper joint)
        self.current_positions[5] = target_pos
        
        # Tell the Action Client (your script) that the movement was successful
        goal_handle.succeed()
        
        # Reply with the final status
        result = GripperCommand.Result()
        result.position = target_pos
        result.reached_goal = True
        return result

    def timer_callback(self):
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = self.joint_names
        js_msg.position = self.current_positions
        self.pub.publish(js_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RVizTranslator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()