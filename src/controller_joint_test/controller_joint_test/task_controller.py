import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from enum import Enum
import math
import numpy as np
import time

# Import your validated mathematical models
from .VisionTransform import transform_camera_to_base
from .ForwardKinematics import forward_kinematics

class RobotState(Enum):
    STARTUP = 0
    SCANNING = 1
    APPROACHING = 2
    WEEDING = 3
    DONE = 4

class PBVSTaskController(Node):
    def __init__(self):
        super().__init__('task_controller')
        
        # State machine initialization
        self.state = RobotState.STARTUP
        self.state_start_time = time.time()
        self.initial_pose_recorded = False

        # Variables to store the latest incoming data
        self.current_joints = None
        self.latest_cam_weed_pos = None  # Will hold [x, y, z] from the camera frame

        # Memory variable to store the exact weed location before plunging
        self.locked_weed_base_pos = None
        
        # 1. PUBLISHER: Send Cartesian coordinates to IK Streamer via Float64MultiArray
        self.target_pub = self.create_publisher(Float64MultiArray, '/desired_tcp_pose_euler', 10)
        
        # 2. SUBSCRIPTION: Listen to the physical robot's joint angles
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # 3. SUBSCRIPTION: Listen to the YOLO/ToF node 
        self.vision_sub = self.create_subscription(
            Point,
            '/weed_location_cam_frame', 
            self.vision_callback,
            10
        )
        
        # The main control loop running at 10 Hz (0.1 seconds) for smooth visual servoing
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("PBVS started! State: SCANNING...")

    def joint_state_callback(self, msg):
        
        # We only expect 5 physical joints from the hardware
        expected_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        
        # Create an array of 6 zeros, so the last (removed) joint 6 is always 0.
        sorted_joints = [0.0] * 6

        # Make sure the message has the joints we care about
        if all(name in msg.name for name in expected_joints):
            for i, expected_name in enumerate(expected_joints):
                # Find exactly where this joint is in the message array
                idx = msg.name.index(expected_name)
                sorted_joints[i] = msg.position[idx]
            
            self.current_joints = sorted_joints

    def vision_callback(self, msg):
        """Stores the latest weed coordinates seen by the camera."""
        self.latest_cam_weed_pos = [msg.x, msg.y, msg.z]

    def control_loop(self):
        """The main State Machine logic."""
        # Safety check: Do nothing until we have received at least one joint state message
        if self.current_joints is None:
            return

        current_time = time.time()
        elapsed_time = current_time - self.state_start_time
        
        if self.state == RobotState.STARTUP:
            # 1. Record the known mechanical Home pose
            if not self.initial_pose_recorded:
                self.start_x = 0.045
                self.start_y = 0.000
                self.start_z = 0.837
                self.start_roll = 0.0
                self.start_yaw = math.pi
                
                self.initial_pose_recorded = True
                self.get_logger().info("Starting smooth 6D interpolation to SCANNING pose...")
            
            # 2. Define exactly where we want to end up (The scanning pose)
            target_x = 0.5
            target_y = 0.0
            target_z = 0.1
            target_roll = math.pi
            target_yaw = 0.0
            
            startup_duration = 5.0  # Take exactly 5 seconds to complete the move
            
            # 3. Calculate a progress percentage from 0.0 to 1.0
            progress = elapsed_time / startup_duration
            
            # 4. When 5 seconds have passed, we are done! Switch to SCANNING
            if progress >= 1.0:
                progress = 1.0
                self.get_logger().info("Reached scan position! Starting sweep.")
                self.state = RobotState.SCANNING
                self.state_start_time = current_time
                self.latest_cam_weed_pos = None  # Clear any old camera noise
                return
                
            # 5. Linear Interpolation (Lerp) for XYZ and Angles
            current_x = self.start_x + (target_x - self.start_x) * progress
            current_y = self.start_y + (target_y - self.start_y) * progress
            current_z = self.start_z + (target_z - self.start_z) * progress
            
            current_roll = self.start_roll + (target_roll - self.start_roll) * progress
            current_yaw = self.start_yaw + (target_yaw - self.start_yaw) * progress
            
            # Stream the smoothly changing coordinates and orientation
            self.publish_target(
                current_x, current_y, current_z, 
                roll=current_roll, yaw=current_yaw
            )

        elif self.state == RobotState.SCANNING:
            if self.latest_cam_weed_pos is not None:
                self.get_logger().info("WEED DETECTED! Switching state: APPROACHING.")
                self.state = RobotState.APPROACHING
                self.state_start_time = current_time
                
        elif self.state == RobotState.APPROACHING:
            # PBVS DYNAMIC LOOK-AND-MOVE LOGIC
            if self.latest_cam_weed_pos is None:
                self.get_logger().warn("Lost sight of weed! Pausing...", throttle_duration_sec=2.0)
                return
            
            # Transform camera pixels to base coordinates continuously
            cam_x, cam_y, cam_z = self.latest_cam_weed_pos
            target_base_pos = transform_camera_to_base(cam_x, cam_y, cam_z, self.current_joints)
            
            # Hover 15 cm directly above the weed
            hover_x = target_base_pos[0]
            hover_y = target_base_pos[1]
            hover_z = target_base_pos[2] + 0.15
            
            # Check if we have physically reached the hover point
            actual_ee_matrix = forward_kinematics(self.current_joints)
            actual_x = actual_ee_matrix[0, 3]
            actual_y = actual_ee_matrix[1, 3]
            actual_z = actual_ee_matrix[2, 3]
            
            distance_to_target = math.sqrt(
                (hover_x - actual_x)**2 + 
                (hover_y - actual_y)**2 + 
                (hover_z - actual_z)**2
            )

            # Dynamic path planner:
            # Generate small intermediate points on a straight line towards the target.
            # Max speed: 10 cm per second (which is 1 cm per 0.1 second tick)
            max_step_per_tick = 0.01 
            
            if distance_to_target > max_step_per_tick:
                # Calculate the straight-line vector to the target
                vector_x = (hover_x - actual_x) / distance_to_target
                vector_y = (hover_y - actual_y) / distance_to_target
                vector_z = (hover_z - actual_z) / distance_to_target
                
                # Take a small step along the vector
                step_x = actual_x + (vector_x * max_step_per_tick)
                step_y = actual_y + (vector_y * max_step_per_tick)
                step_z = actual_z + (vector_z * max_step_per_tick)
                
                self.publish_target(step_x, step_y, step_z)
            else:
                # We are close enough to just snap to the final hover position
                self.publish_target(hover_x, hover_y, hover_z)
            
            if distance_to_target < 0.015:
                self.get_logger().info("Perfectly aligned over weed! Locking target and switching to WEEDING.")
                
                # Lock the exact absolute coordinates of the weed in memory
                self.locked_weed_base_pos = target_base_pos
                self.weeding_start_z = actual_z

                self.state = RobotState.WEEDING
                self.state_start_time = current_time
        
        elif self.state == RobotState.WEEDING: 
            # Retrieve the locked coordinates of the weed
            target_x = self.locked_weed_base_pos[0]
            target_y = self.locked_weed_base_pos[1]
            target_z = self.locked_weed_base_pos[2] - 0.04 # Plunge 4 cm below the plant
            
            plunge_duration = 2.0  # Take exactly 2 seconds to plunge down smoothly
            progress = elapsed_time / plunge_duration
            
            if progress <= 1.0:
                # Cartesian path planner:
                # Interpolate the Z axis to create a perfectly straight vertical line
                current_z = self.weeding_start_z + (target_z - self.weeding_start_z) * progress
                
                self.publish_target(target_x, target_y, current_z)
            else:
                # We have reached the bottom! Hold the position
                self.publish_target(target_x, target_y, target_z)

            # TODO: Close the gripper here!

            # Wait 1.5 seconds for the gripper physical mechanism to close
            if elapsed_time > (plunge_duration + 1.5):
                self.get_logger().info("Weed grabbed! Switching to DONE.")
                self.state = RobotState.DONE
                self.state_start_time = current_time          
                   
        elif self.state == RobotState.DONE:
            # Pull the arm straight back up to a safe position
            self.publish_target(0.4, 0.0, 0.1)
            self.get_logger().info("Task complete. Waiting in safe position.", throttle_duration_sec=2.0)
            self.latest_cam_weed_pos = None

    def publish_target(self, x, y, z, roll=np.pi, pitch=0, yaw=None):
        """
        Packages the XYZ coordinates and Euler angles into a Float64MultiArray.
        Defaults to pointing straight down (roll = pi) unless told otherwise.
        If yaw is not explicitly provided, it forces the wrist (Joint 4 and 6) to stay neutral by aligning Yaw with the base.
        """
        msg = Float64MultiArray()
        
        if yaw is None:
            yaw = np.arctan2(y,x)
        
        # Package exactly as [X, Y, Z, R, P, Y]
        msg.data = [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]
        
        self.target_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PBVSTaskController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()