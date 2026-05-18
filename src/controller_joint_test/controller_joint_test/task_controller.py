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
    STARTUP = 1
    SCANNING = 2
    APPROACHING = 3
    CENTERING_XY = 4
    SECOND_SCAN = 5
    PLUNGING = 6
    GRASPING = 7
    WEEDING = 8
    DONE = 9
    DYNAMIC_APPROACHING = 10
    PAUSE = 11


class PBVSTaskController(Node):
    def __init__(self):
        super().__init__('task_controller')
        
        # State machine initialization
        self.initial_pose_recorded = False
        self.start_scanning = False
        self.state_start_time = time.time()
        self.state = RobotState.STARTUP

        # --- TUNABLE PARAMETERS (Configuration) ---
        # The acceptable 3D Euclidean error (in meters) to consider a target reached
        self.target_tolerance_m = 0.01
        
        # The vertical distance (in meters) to hover above the weed before plunging
        self.hover_offset_z_m = 0.10

        # The vertical distance (in meters) to plunge into the ground
        self.plunge_depth_m = 0.04
        
        # The default speed of the TCP for interpolated movements
        self.tcp_speed_m_s = 0.08

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
        
        if self.state == RobotState.STARTUP:
            if not self.initial_pose_recorded:
                # Open the gripper at startup:
                self.set_gripper('open')
                
                # Define the scanning pose
                self.scan_x = 0.5
                self.scan_y = 0.0
                self.scan_z = 0.1
                target_roll = math.pi
                target_pitch = 0.0
                target_yaw = 0.0
            
                self.publish_target(
                    self.scan_x, self.scan_y, self.scan_z, 
                    roll=target_roll, pitch=target_pitch, yaw=target_yaw
                )

                self.initial_pose_recorded = True

            # Calculate where the end-effector is right now
            actual_pos = self.get_current_tcp()
            error = self.calculate_tcp_error([self.scan_x, self.scan_y , self.scan_z], actual_pos)
                
            # 4. Switch to SCANNING when scanning pose is reached
            if error <= self.target_tolerance_m:
                self.switch_state(RobotState.SCANNING)
                self.latest_cam_weed_pos = None  # Clear any old camera noise
                return

        elif self.state == RobotState.SCANNING:
            if not self.start_scanning:
                self.get_logger().info("Sent trigger to vision detection")
                vision_detection = Int32()
                vision_detection.data = 30
                self.vision_command.publish(vision_detection)
                self.start_scanning = True
            if self.latest_cam_weed_pos is not None:
                self.switch_state(RobotState.CENTERING_XY)

        elif self.state == RobotState.CENTERING_XY:
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
            target_y = self.locked_weed_base_pos[1] + 0.051 + 0.02
            target_z = self.approach_start_z
            
            distance_to_target = self.calculate_tcp_error([target_x, target_y , target_z], [self.approach_start_x, self.approach_start_y, self.approach_start_z])
            move_duration = distance_to_target / self.tcp_speed_m_s
            move_duration = max(0.2, move_duration)
            progress = elapsed_time / move_duration
            
            if progress <= 1.0:
                # Cartesian Path Planner: Interpolate smoothly to the target
                current_x = self.interpolate_value(self.approach_start_x, target_x, progress)
                current_y = self.interpolate_value(self.approach_start_y, target_y, progress)
                current_z = self.approach_start_z
                
                # Orientation remains pointing straight down (handled by default in publish_target)
                self.publish_target(current_x, current_y, current_z)
            else:
                # Hover point reached
                self.publish_target(target_x, target_y, target_z)
                self.start_scanning = False
                self.latest_cam_weed_pos = None
                self.locked_weed_base_pos = None
                self.switch_state(RobotState.SECOND_SCAN)

        elif self.state == RobotState.SECOND_SCAN:
            if not self.start_scanning:
                self.get_logger().info("Sent trigger to vision detection")
                vision_detection = Int32()
                vision_detection.data = 30
                self.vision_command.publish(vision_detection)
                self.start_scanning = True
            if self.latest_cam_weed_pos is not None:
                self.switch_state(RobotState.APPROACHING)

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
            target_z = self.locked_weed_base_pos[2] + self.hover_offset_z_m
            
            distance_to_target = self.calculate_tcp_error([target_x, target_y , target_z], [self.approach_start_x, self.approach_start_y, self.approach_start_z])
            move_duration = distance_to_target / self.tcp_speed_m_s
            move_duration = max(0.2, move_duration)
            progress = elapsed_time / move_duration
            
            if progress <= 1.0:
                # Cartesian Path Planner: Interpolate smoothly to the target
                current_x = self.interpolate_value(self.approach_start_x, target_x, progress)
                current_y = self.interpolate_value(self.approach_start_y, target_y, progress)
                current_z = self.interpolate_value(self.approach_start_z, target_z, progress)
                
                # Orientation remains pointing straight down (handled by default in publish_target)
                self.publish_target(current_x, current_y, current_z)
            else:
                # Hover point reached
                self.publish_target(target_x, target_y, target_z)
                # Record the Z-height for the weeding plunge
                self.weeding_start_z = target_z
                self.switch_state(RobotState.PLUNGING)

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
            hover_z = target_base_pos[2] + self.hover_offset_z_m
            
            # Check if we have physically reached the hover point
            actual_x, actual_y, actual_z = self.get_current_tcp()
            distance_to_target = self.calculate_tcp_error([hover_x, hover_y , hover_z], [actual_x, actual_y, actual_z])

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
            
            if distance_to_target < self.target_tolerance_m:
                # Lock the exact absolute coordinates of the weed in memory
                self.locked_weed_base_pos = target_base_pos
                self.weeding_start_z = actual_z

                self.switch_state(RobotState.PLUNGING)
        
        elif self.state == RobotState.PLUNGING:
            # Retrieve the locked coordinates of the weed
            self.target_x = self.locked_weed_base_pos[0]
            self.target_y = self.locked_weed_base_pos[1]
            self.target_z = self.locked_weed_base_pos[2] - self.plunge_depth_m # Plunge below the plant
            
            distance_to_target = abs(self.target_z - self.weeding_start_z)
            move_duration = distance_to_target / self.tcp_speed_m_s
            move_duration = max(0.2, move_duration)
            progress = elapsed_time / move_duration
            
            if progress <= 1.0:
                # Cartesian path planner:
                # Interpolate the Z axis to create a perfectly straight vertical line
                current_z = self.interpolate_value(self.weeding_start_z, self.target_z, progress)
                self.publish_target(self.target_x, self.target_y, current_z)
            else:
                # We have reached the bottom! Hold the position
                self.publish_target(self.target_x, self.target_y, self.target_z)
            
            # Check if we have physically reached the plunged position
            actual_x, actual_y, actual_z = self.get_current_tcp()
            error = self.calculate_tcp_error([self.target_x, self.target_y, self.target_z], [actual_x, actual_y, actual_z])

            if error <= self.target_tolerance_m:
                # Close the gripper:
                self.set_gripper('close')
                self.switch_state(RobotState.GRASPING)
  
        elif self.state == RobotState.GRASPING:
            if elapsed_time > 2.0: # Waiting for the gripper to physically close
                self.switch_state(RobotState.WEEDING)
        
        elif self.state == RobotState.WEEDING:
            distance_to_target = abs(self.target_z - self.weeding_start_z)
            move_duration = distance_to_target / self.tcp_speed_m_s
            move_duration = max(0.2, move_duration)
            progress = elapsed_time / move_duration
            if progress <= 1.0:
                # Pull straight up
                # Cartesian path planner: Interpolate the Z axis to create a perfectly straight vertical line
                current_z = self.interpolate_value(self.target_z, self.weeding_start_z, progress)
                self.publish_target(self.target_x, self.target_y, current_z)
            else:
                self.publish_target(self.target_x, self.target_y, self.weeding_start_z)
                self.switch_state(RobotState.DONE) 

        elif self.state == RobotState.DONE:
            distance_to_target = self.calculate_tcp_error([self.target_x, self.target_y , self.weeding_start_z], [self.scan_x, self.scan_y, self.scan_z])
            move_duration = distance_to_target / self.tcp_speed_m_s
            move_duration = max(0.2, move_duration)
            progress = elapsed_time / move_duration
            if progress <= 1.0:
                # Cartesian Path Planner: Interpolate smoothly to the target
                current_x = self.interpolate_value(self.target_x, self.scan_x, progress)
                current_y = self.interpolate_value(self.target_y, self.scan_y, progress)
                current_z = self.interpolate_value(self.weeding_start_z, self.scan_z, progress)
                self.publish_target(current_x, current_y, current_z)
            else:
                self.publish_target(self.scan_x, self.scan_y, self.scan_z)
                self.get_logger().info("Task complete. Waiting in safe position.", throttle_duration_sec=2.0)
                #Open the gripper
                self.set_gripper('open')
                self.start_scanning = False
                self.latest_cam_weed_pos = None
                self.locked_weed_base_pos = None
                self.switch_state(RobotState.PAUSE)

        elif self.state == RobotState.PAUSE:
            if elapsed_time > 10.0: 
                self.switch_state(RobotState.STARTUP)

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
    
    def calculate_tcp_error(self, target_pos, actual_pos):
        """
        Calculates the Euclidean distance (3D error) between the target and actual TCP positions.
        """
        tx, ty, tz = target_pos
        ax, ay, az = actual_pos
        return math.sqrt((tx - ax)**2 + (ty - ay)**2 + (tz - az)**2)
    
    def switch_state(self, new_state):
        """
        Handles transitioning the robot to a new state.
        Updates the state variable, resets the internal state timer, 
        and logs the transition to the terminal.
        """
        self.state = new_state
        self.state_start_time = time.time()
        
        # Log the state change dynamically using the Enum's string name
        self.get_logger().info(f"--- Transitioning to State: {new_state.name} ---")

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