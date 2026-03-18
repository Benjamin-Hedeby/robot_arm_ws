import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float64MultiArray

class ArmNode(Node):
    def __init__(self):
        super().__init__('arm_node')
        
        # Subscriber 1: Joy Stick
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)

        # Subscriber 2: Current Joint Positions
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)

        # Publisher: Arm Commands
        self.arm_pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)

        # Internal state to store latest data
        self.current_joint_positions = [0.0] * 6 

        self.has_received_joints = False

    # Function called when /joint_states recieved
    def joint_state_callback(self, msg):
        
        expected_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        # Make sure the message has the joints we care about
        if all(name in msg.name for name in expected_joints):
            for i, expected_name in enumerate(expected_joints):
                # Find exactly where this joint is in the message array
                idx = msg.name.index(expected_name)
                self.current_joint_positions[i] = msg.position[idx]
            
            # Allow joy commands now that we know exactly where the arm is
            self.has_received_joints = True

    def joy_callback(self, msg):

        if not self.has_received_joints:
            return
        
        cmd = Float64MultiArray()

        target_positions = self.current_joint_positions.copy()

        # Deadzone for joystick to prevent drift
        deadzone = 0.05
        max_step_odrive = 0.1
        max_step_servo = 0.1
        command_set = False

        # Joystick axis 0 controls joint0, axis 1 controls joint1, axis 3 controls joint3, axis 4 controls joint2
        if abs(msg.axes[0]) > deadzone:

            target_positions[0] -= max_step_odrive * msg.axes[0]

            command_set = True

        if abs(msg.axes[1]) > deadzone:

            target_positions[1] += max_step_odrive * msg.axes[1]

            command_set = True

        if abs(msg.axes[3]) > deadzone:

            target_positions[3] -= max_step_servo * msg.axes[3]

            command_set = True

        if abs(msg.axes[4]) > deadzone:

            target_positions[2] -= max_step_odrive * msg.axes[4]

            command_set = True

        # Homing by triangle
        if msg.buttons[2] == 1:

            target_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            # 6. Put the Point inside the main Command 
            cmd.data = target_positions

            self.arm_pub.publish(cmd)

            return

        if command_set:
            
            # 6. Put the Point inside the main Command 
            cmd.data = target_positions

            self.arm_pub.publish(cmd)
    

def main(args=None):
    rclpy.init(args=args)
    node = ArmNode()
    rclpy.spin(node) # Keeps the node alive
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()