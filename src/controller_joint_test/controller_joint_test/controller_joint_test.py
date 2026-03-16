import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration  

class ArmNode(Node):
    def __init__(self):
        super().__init__('minimal_arm_node')
        
        # Subscriber 1: Joy Stick
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)

        # Subscriber 2: Current Joint Positions
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)

        # Publisher: Arm Commands
        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)

        # Internal state to store latest data
        self.current_joint_positions = [0.0] * 6 

    # Function called when /joint_states recieved
    def joint_state_callback(self, msg):
        if len(msg.position) >= 6:
            self.current_joint_positions = list(msg.position[:6])

    def joy_callback(self, msg):
        cmd = JointTrajectory()
        cmd.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        # 1. Create the specific "Point" doll
        point = JointTrajectoryPoint()

        # 2. Copy our current positions so we don't snap everything else to zero!
        target_positions = self.current_joint_positions.copy()

        # 3. Modify joint3 (index 2) if the button is pressed
        if msg.buttons[1] == 1:
            target_positions[2] += 0.05  # Add a tiny step (about 2.8 degrees)

        # 4. Put the positions array into the Point doll
        point.positions = target_positions
        
        # 5. Tell the controller how fast to get there (0.1 seconds)
        point.time_from_start = Duration(sec=0.1, nanosec=0)

        # 6. Put the Point inside the main Command 
        cmd.points = [point]

        self.arm_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ArmNode()
    rclpy.spin(node) # Keeps the node alive
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()