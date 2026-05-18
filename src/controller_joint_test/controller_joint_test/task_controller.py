import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String, Int32
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from enum import Enum
import math
import numpy as np
import time
from .VisionTransform import transform_camera_to_base
from .ForwardKinematics import forward_kinematics

class RobotState(Enum):
    STARTUP_INTERPOLATION = 0
    STARTUP_SIMPLE = 1
    SCANNING = 2
    APPROACHING = 3
    WEEDING = 4
    DONE = 5
    DYNAMIC_APPROACHING = 6
    PAUSE = 7
    SIMPLE_APPROACHING = 8
    FIRST_SCAN = 9

class PBVSTaskController(Node):
    def __init__(self):
        super().__init__('task_controller')
        
        # State machine initialization
        self.initial_pose_recorded = False
        self.start_scanning = False
        self.state_start_time = time.time()
        self.state = RobotState.STARTUP_SIMPLE

        # Variables to store the latest incoming data
        self.current_joints = None       # Will hold [q_1,..., q_5] from the motor encoders
        self.latest_cam_weed_pos = None  # Will hold [x, y, z] from the camera frame
        self.locked_weed_base_pos = None # Memory variable to store the exact weed location before plunging
        
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

        # 4. PUBLISHER: Send command to gripper
        self.gripper_command = self.create_publisher(String, '/gripper_open_close_cmd', 10)

        # 5. PUBLISHER: Send command to vision detection
        self.vision_command = self.create_publisher(Int32, '/trigger_measurement', 10)
        
        # The main control loop running at 100 Hz (0.01 seconds) for smooth visual servoing
        self.timer = self.create_timer(0.01, self.control_loop)
        self.get_logger().info("PBVS started! State: STARTUP...")

    def joint_state_callback(self, msg):       
        # We expect 5 physical joints from the hardware
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
        """The main Finite State Machine logic."""
        # Safety check: Do nothing until we have received at least one joint state message
        if self.current_joints is None:
            self.get_logger().info(f"Waiting untill we have received joint states", throttle_duration_sec=2.0)
            return

        current_time = time.time()
        elapsed_time = current_time - self.state_start_time
        
        if self.state == RobotState.STARTUP_SIMPLE:
            if not self.initial_pose_recorded:
                # Open the gripper at startup:
                self.set_gripper('open')
                
                # Define the scanning pose
                self.target_x = 0.5
                self.target_y = 0.0
                self.target_z = 0.1
                target_roll = math.pi
                target_pitch = 0.0
                target_yaw = 0.0
            
                self.publish_target(
                    self.target_x, self.target_y, self.target_z, 
                    roll=target_roll, pitch=target_pitch, yaw=target_yaw
                )

                self.initial_pose_recorded = True

            # Calculate where the end-effector is right now
            actual_x, actual_y, actual_z = self.get_current_tcp()

            error = abs(self.target_x-actual_x) + abs(self.target_y-actual_y) + abs(self.target_z - actual_z)
                
            # 4. Switch to SCANNING when scanning pose is reached
            if error <= 0.05:
                self.get_logger().info("Reached scan position! Starting sweep.")
                self.state = RobotState.SCANNING
                self.state_start_time = current_time
                self.latest_cam_weed_pos = None  # Clear any old camera noise
                return

        if self.state == RobotState.STARTUP_INTERPOLATION:
            # 1. Record the known mechanical Home pose
            if not self.initial_pose_recorded:
                self.start_x = 0.045
                self.start_y = 0.000
                self.start_z = 0.837
                self.start_roll = math.pi
                self.start_pitch = -math.pi
                self.start_yaw = 0.0
                
                # Open the gripper at startup:
                self.set_gripper('open')

                self.initial_pose_recorded = True
                self.get_logger().info("Starting smooth interpolation to SCANNING pose...")
            
            # 2. Define the scanning pose
            target_x = 0.5
            target_y = 0.0
            target_z = 0.1
            target_roll = math.pi
            target_pitch = 0.0
            target_yaw = 0.0
            
            startup_duration = 5.0
            
            # 3. Calculate a progress percentage from 0.0 to 1.0
            progress = elapsed_time / startup_duration
            
            # 4. Switch to SCANNING when scanning pose is reached
            if progress >= 1.0:
                progress = 1.0
                self.get_logger().info("Reached scan position! Starting sweep.")
                self.state = RobotState.SCANNING
                self.state_start_time = current_time
                self.latest_cam_weed_pos = None  # Clear any old camera noise
                return
                
            # 5. Linear Interpolation for XYZ and Angles
            current_x = self.start_x + (target_x - self.start_x) * progress
            current_y = self.start_y + (target_y - self.start_y) * progress
            current_z = self.start_z + (target_z - self.start_z) * progress
            
            current_roll = self.start_roll + (target_roll - self.start_roll) * progress
            current_pitch = self.start_pitch + (target_pitch - self.start_pitch) * progress
            current_yaw = self.start_yaw + (target_yaw - self.start_yaw) * progress
            
            # Stream the smoothly changing coordinates and orientation
            self.publish_target(
                current_x, current_y, current_z, 
                roll=current_roll, pitch=current_pitch, yaw=current_yaw
            )

        elif self.state == RobotState.SCANNING:
            if not self.start_scanning:
                self.get_logger().info("Sent trigger to vision detection")
                vision_detection = Int32()
                vision_detection.data = 30
                self.vision_command.publish(vision_detection)
                self.start_scanning = True
            if self.latest_cam_weed_pos is not None:
                self.get_logger().info("WEED DETECTED! Switching state: APPROACHING.")
                self.state = RobotState.APPROACHING
                self.state_start_time = current_time

        elif self.state == RobotState.SIMPLE_APPROACHING:
            # LOOK-THEN-MOVE LOGIC
            # Lock the target coordinate
            if self.locked_weed_base_pos is None:
                if self.latest_cam_weed_pos is None:
                    self.get_logger().warn("Waiting for camera data before moving...", throttle_duration_sec=2.0)
                    return
                # Transform camera pixels to base coordinates
                cam_x, cam_y, cam_z = self.latest_cam_weed_pos
                self.locked_weed_base_pos = transform_camera_to_base(cam_x, cam_y, cam_z, self.current_joints)
                
                self.get_logger().info(f"Target locked at Base Pos: [{self.locked_weed_base_pos[0]:.3f}, {self.locked_weed_base_pos[1]:.3f}, {self.locked_weed_base_pos[2]:.3f}]")
                self.get_logger().info("Starting 'blind' move to hover position.")
                # Hover above the weed
                self.target_x = self.locked_weed_base_pos[0]
                self.target_y = self.locked_weed_base_pos[1] 
                self.target_z = self.locked_weed_base_pos[2] + 0.15
                self.publish_target(self.target_x, self.target_y, self.target_z)

            # Calculate where the end-effector is right now
            actual_x, actual_y, actual_z = self.get_current_tcp()

            error = abs(self.target_x-actual_x) + abs(self.target_y-actual_y) + abs(self.target_z - actual_z)
                
            # 4. Switch to SCANNING when scanning pose is reached
            if error <= 0.05:
                self.get_logger().info("Hover position reached. Switching to WEEDING.")

                # Record the Z-height for the weeding plunge
                self.weeding_start_z = target_z
                
                self.state = RobotState.PAUSE
                self.state_start_time = current_time
                return

        elif self.state == RobotState.APPROACHING:
            # LOOK-THEN-MOVE LOGIC
            # --- PHASE 1: THE "LOOK" ---
            # Lock the target coordinate
            if self.locked_weed_base_pos is None:
                if self.latest_cam_weed_pos is None:
                    self.get_logger().warn("Waiting for camera data before moving...", throttle_duration_sec=2.0)
                    return
                # Transform camera pixels to base coordinates
                cam_x, cam_y, cam_z = self.latest_cam_weed_pos
                self.locked_weed_base_pos = transform_camera_to_base(cam_x, cam_y, cam_z, self.current_joints)
                
                # Calculate where the end-effector is right now (the start of the move)
                self.approach_start_x, self.approach_start_y, self.approach_start_z = self.get_current_tcp()
                
                self.get_logger().info(f"Target locked at Base Pos: [{self.locked_weed_base_pos[0]:.3f}, {self.locked_weed_base_pos[1]:.3f}, {self.locked_weed_base_pos[2]:.3f}]")
                self.get_logger().info("Starting 'blind' move to hover position.")

            # --- PHASE 2: THE "MOVE" ---
            # Hover above the weed
            target_x = self.locked_weed_base_pos[0]
            target_y = self.locked_weed_base_pos[1] 
            target_z = self.locked_weed_base_pos[2] + 0.15
            
            approach_duration = 4.0
            progress = elapsed_time / approach_duration
            
            if progress <= 1.0:
                # Cartesian Path Planner: Interpolate smoothly to the target
                current_x = self.approach_start_x + (target_x - self.approach_start_x) * progress
                current_y = self.approach_start_y + (target_y - self.approach_start_y) * progress
                current_z = self.approach_start_z + (target_z - self.approach_start_z) * progress
                
                # Orientation remains pointing straight down (handled by default in publish_target)
                self.publish_target(current_x, current_y, current_z)
            else:
                # Hover point reached
                self.publish_target(target_x, target_y, target_z)
                self.get_logger().info("Hover position reached. Switching to WEEDING.")
                
                # Record the Z-height for the weeding plunge
                self.weeding_start_z = target_z
                
                self.state = RobotState.WEEDING
                self.state_start_time = current_time

        elif self.state == RobotState.DYNAMIC_APPROACHING:
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
            actual_x, actual_y, actual_z = self.get_current_tcp()
            
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
            self.target_x = self.locked_weed_base_pos[0]
            self.target_y = self.locked_weed_base_pos[1]
            self.target_z = self.locked_weed_base_pos[2] - 0.04 # Plunge 4 cm below the plant
            
            plunge_duration = 3.0
            progress = elapsed_time / plunge_duration
            
            if progress <= 1.0:
                # Cartesian path planner:
                # Interpolate the Z axis to create a perfectly straight vertical line
                current_z = self.weeding_start_z + (self.target_z - self.weeding_start_z) * progress
                
                self.publish_target(self.target_x, self.target_y, current_z)
            else:
                # We have reached the bottom! Hold the position
                self.publish_target(self.target_x, self.target_y, self.target_z)
                # Close the gripper:
                self.set_gripper('close')

            # Wait for the gripper physical mechanism to close
            if elapsed_time > (plunge_duration + 3.0):
                self.get_logger().info("Weed grabbed! Switching to DONE.")
                self.state = RobotState.DONE
                self.state_start_time = current_time          
                   
        elif self.state == RobotState.DONE:
            plunge_duration = 3.0
            progress = elapsed_time / plunge_duration
            if progress <= 1.0: 
                # Pull straight up
                # Cartesian path planner: Interpolate the Z axis to create a perfectly straight vertical line
                current_z = self.target_z + (self.weeding_start_z - self.target_z) * progress
                
                self.publish_target(self.target_x, self.target_y, current_z)
            elif progress > 1.0 and progress <=2.0:
                # Go back to scanning pose
                # Cartesian Path Planner: Interpolate smoothly
                current_x = self.target_x + (0.5 - self.target_x) * (progress-1.0)
                current_y = self.target_y + (0.0 - self.target_y) * (progress-1.0)
                current_z = self.weeding_start_z + (0.1 - self.weeding_start_z) * (progress-1.0)
                self.publish_target(current_x, current_y, current_z)
            else:
                self.publish_target(0.5, 0.0, 0.1)
                self.get_logger().info("Task complete. Waiting in safe position.", throttle_duration_sec=2.0)
                #Open the gripper
                self.set_gripper('open')
                self.latest_cam_weed_pos = None
                self.state = RobotState.PAUSE
                self.state_start_time = current_time

        elif self.state == RobotState.PAUSE:
            time.sleep(1000)

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
    
    def set_gripper(self, state_str):
        """
        Publishes the command to open or close the gripper.
        :param state_str: String containing 'open' or 'close'
        """
        gripper_state = String()
        gripper_state.data = state_str
        self.gripper_command.publish(gripper_state)
    
    def get_current_tcp(self):
        """
        Calculates and returns the current TCP (End-Effector) position [x, y, z].
        """
        actual_ee_matrix = forward_kinematics(self.current_joints)
        actual_x = actual_ee_matrix[0, 3]
        actual_y = actual_ee_matrix[1, 3]
        actual_z = actual_ee_matrix[2, 3]
        
        return actual_x, actual_y, actual_z
    
    def interpolate_value(self, start_val, target_val, progress):
        """
        Calculates the linear interpolation between a start and target value 
        based on the current progress (0.0 to 1.0).
        """
        return start_val + (target_val - start_val) * progress

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