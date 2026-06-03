import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from control_msgs.action import GripperCommand
from .configuration import GRIPPER_CLOSED_POSITION, GRIPPER_OPEN_POSITION, GRIPPER_MAX_EFFORT

class GripperTranslatorNode(Node):
    def __init__(self):
        super().__init__('gripper_translator_node')
        
        # 1. The Action Client (Talks to the physical gripper)
        self.gripper_client = ActionClient(
            self, 
            GripperCommand, 
            '/gripper_action_controller/gripper_cmd'
        )
        
        # 2. The Subscriber (Listens for your text commands)
        self.command_sub = self.create_subscription(
            String, 
            '/gripper_open_close_cmd', 
            self.command_callback, 
            10
        )
        
        self.get_logger().info('Gripper Translator ready! Publish "open" or "close" to /gripper_simple_cmd')

    def command_callback(self, msg):
        # Convert the incoming text to lowercase and remove accidental spaces
        command = msg.data.strip().lower()
        
        if command == 'open':
            self.get_logger().info('Received "open". Opening gripper...')
            self.send_goal(GRIPPER_OPEN_POSITION)
            
        elif command == 'close':
            self.get_logger().info('Received "close". Closing gripper...')
            self.send_goal(GRIPPER_CLOSED_POSITION)
            
        else:
            self.get_logger().warn(f'Unknown command: "{command}". Please send "open" or "close".')

    def send_goal(self, target_position):
        # Wait up to 1 second for the hardware to be ready
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Gripper action server is offline! Is the Pi running?')
            return
            
        # Build the action goal
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = target_position
        goal_msg.command.max_effort = GRIPPER_MAX_EFFORT
        
        self.gripper_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GripperTranslatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()