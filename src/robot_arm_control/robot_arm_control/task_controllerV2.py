import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String, Int32
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState, Imu
from enum import Enum
from collections import deque
import math
import csv
import os
import time
import sys
import select
from .configuration import CAMERA_OFFSET_Y
from .VisionTransformV2 import transform_camera_to_base
from .ForwardKinematics import forward_kinematics

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
 
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
 
    return roll_x, pitch_y, yaw_z

class RobotState(Enum):
    STARTUP = 1
    SCAN_SWEEP = 2
    SCAN_CONFIRM = 3
    ALIGNING_XY = 4
    FINAL_SCAN = 5
    APPROACHING = 6
    PLUNGING = 7
    GRASPING = 8
    EXTRACTING = 9
    DONE = 10
    PAUSE = 11

class PBVSTaskController(Node):
    # Vision Trigger Constants
    TRIGGER_SWEEP = 2
    TRIGGER_CONFIRM = 50

    def __init__(self):
        super().__init__('task_controller')
        
        # State machine initialization
        self.initial_pose_recorded = False
        self.start_scanning = False
        self.state_start_time = time.time()
        self.state = RobotState.PAUSE

        # --- TUNABLE PARAMETERS (Configuration) ---
        self.target_tolerance_m = 0.01      # Acceptable 3D Euclidean error (m) to consider a target reached
        self.hover_offset_z_m = 0.10        # Vertical hover distance (m)
        self.plunge_depth_m = 0.04          # Vertical plunge depth into ground (m)
        self.tcp_speed_m_s = 0.08           # Default speed of the TCP (m/s)

        # --- 90-DEGREE ARC SWEEP PARAMETERS ---
        self.sweep_radius_m = 0.50              # Radius of the arc (m)
        self.angle_start_rad = -math.pi / 6.0   # Starting angle
        self.angle_end_rad =  math.pi / 6.0     # End angle
        self.sweep_period_s = 20.0              # Seconds to complete one full round-trip
        self.scan_z = 0.10                      # Height of TCP while scanning              

        # --- HARDWARE STATE VARIABLES ---
        self.current_joints = None       # Will hold [q_1,..., q_5] from the motor encoders
        self.latest_cam_weed_pos = None  # Will hold [x, y, z] from the camera frame
        self.locked_weed_base_pos = None # Memory variable to store the exact weed location before plunging

        self.camera_roll = None
        self.camera_pitch = None
        self.imu_buffer_size = 5
        self.roll_buffer = deque(maxlen=self.imu_buffer_size)
        self.pitch_buffer = deque(maxlen=self.imu_buffer_size)
        
        # --- ROS2 COMMUNICATION INTERFACES ---
        # PUBLISHER: Send Cartesian coordinates to IK Streamer via Float64MultiArray
        self.target_pub = self.create_publisher(Float64MultiArray, '/desired_tcp_pose_euler', 10)
        
        # PUBLISHER: Send command to gripper
        self.gripper_command = self.create_publisher(String, '/gripper_open_close_cmd', 10)

        # PUBLISHER: Send command to vision detection
        self.vision_command = self.create_publisher(Int32, '/trigger_measurement', 10)

        # SUBSCRIPTION: Listen to the physical robot's joint angles
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        # SUBSCRIPTION: Listen to the vision node 
        self.vision_sub = self.create_subscription(Point, '/weed_location_cam_frame', self.vision_callback, 10)

        # SUBSCRIPTION to the camera IMU
        self.imu_sub = self.create_subscription(Imu, '/oak/imu/data', self.imu_callback, 10)

        # --- CSV LOGGING INITIALIZATION ---
        self.csv_filename = 'weed_locations_base_frame_10.csv'
        file_exists = os.path.isfile(self.csv_filename)
        self.csv_file = open(self.csv_filename, mode='a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            self.csv_writer.writerow(['Timestamp', 'X_Base', 'Y_Base', 'Z_Base', 'State_Context', 'Status'])
            self.csv_file.flush()

        self.prompted_for_input = False
        self.latest_status = "Success" 

        # The main control loop running at 100 Hz (0.01 seconds)
        self.timer = self.create_timer(0.01, self.control_loop)
        self.get_logger().info("Task Controller started! State: STARTUP...")

    # ==========================================
    #             ROS2 CALLBACKS
    # ==========================================
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

    def imu_callback(self, msg):
        """Converts IMU quaternion to Roll, Pitch, Yaw and stores them."""
        # Determine euler angles from the quaternion
        raw_roll, raw_pitch, raw_yaw = euler_from_quaternion(
            msg.orientation.x, 
            msg.orientation.y, 
            msg.orientation.z, 
            msg.orientation.w
        )

        # Axis' on the camera are reversed
        corrected_roll = raw_pitch
        corrected_pitch = raw_roll
        
        # Add the new values to the rolling buffers
        self.roll_buffer.append(corrected_roll)
        self.pitch_buffer.append(corrected_pitch)
        
        # Calculate the average of the buffer and store it
        self.camera_roll = sum(self.roll_buffer) / len(self.roll_buffer)
        self.camera_pitch = sum(self.pitch_buffer) / len(self.pitch_buffer)

    # ==========================================
    #             HELPER METHODS
    # ==========================================
    def log_weed_to_csv(self, position, context_name, status="Success"):
        """Write weed coordinate to a CSV-file."""
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            x, y, z = position
            self.csv_writer.writerow([timestamp, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", context_name, status])
            self.csv_file.flush()
            self.get_logger().info(f"Logged to CSV: [{x:.3f}, {y:.3f}, {z:.3f}] ({context_name}) - Status: {status}")
        except Exception as e:
            self.get_logger().error(f"Failed to write to CSV: {str(e)}")

    def trigger_vision(self, trigger_value: int) -> bool:
        """Handles the logic of triggering the camera and waiting for subscribers."""
        if self.vision_command.get_subscription_count() > 0:
            msg = Int32()
            msg.data = trigger_value
            self.vision_command.publish(msg)
            self.start_scanning = True
            return True
        else:
            self.get_logger().warn("Waiting for camera subscriber on /trigger_measurement...", throttle_duration_sec=2.0)
            return False

    def reset_vision_flags(self):
        """Clears vision data to prepare for the next state's scanning needs."""
        self.start_scanning = False
        self.latest_cam_weed_pos = None
        self.locked_weed_base_pos = None

    def publish_target(self, x, y, z, roll=math.pi, pitch=0, yaw=None):
        """
        Packages the XYZ coordinates and Euler angles into a Float64MultiArray.
        Defaults to pointing straight down (roll = pi) unless told otherwise.
        If yaw is not explicitly provided, it forces the wrist (Joint 4 and 6) to stay neutral by aligning Yaw with the base.
        """
        msg = Float64MultiArray()
        if yaw is None:
            yaw = math.atan2(y, x)
        msg.data = [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]
        self.target_pub.publish(msg)
    
    def set_gripper(self, state_str):
        """
        Publishes the command to open or close the gripper.
        :param state_str: String containing 'open' or 'close'
        """
        msg = String()
        msg.data = state_str
        self.gripper_command.publish(msg)
    
    def get_current_tcp(self):
        """Calculates and returns the current TCP (End-Effector) position [x, y, z]."""
        actual_ee_matrix = forward_kinematics(self.current_joints)
        return actual_ee_matrix[0, 3], actual_ee_matrix[1, 3], actual_ee_matrix[2, 3]
    
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
        self.get_logger().info(f"--- Transitioning to State: {new_state.name} ---")

    # ==========================================
    #          FINITE STATE MACHINE ROUTER
    # ==========================================
    def control_loop(self):
        if self.current_joints is None:
            self.get_logger().info("Waiting until we receive joint states...", throttle_duration_sec=2.0)
            return

        elapsed_time = time.time() - self.state_start_time

        # FSM Routing
        if self.state == RobotState.STARTUP:
            self.handle_startup(elapsed_time)
        elif self.state == RobotState.SCAN_SWEEP:
            self.handle_scan_sweep(elapsed_time)
        elif self.state == RobotState.SCAN_CONFIRM:
            self.handle_scan_confirm(elapsed_time)
        elif self.state == RobotState.ALIGNING_XY:
            self.handle_aligning_xy(elapsed_time)
        elif self.state == RobotState.FINAL_SCAN:
            self.handle_final_scan(elapsed_time)
        elif self.state == RobotState.APPROACHING:
            self.handle_approaching(elapsed_time)
        elif self.state == RobotState.PLUNGING:
            self.handle_plunging(elapsed_time)
        elif self.state == RobotState.GRASPING:
            self.handle_grasping(elapsed_time)
        elif self.state == RobotState.EXTRACTING:
            self.handle_extracting(elapsed_time)
        elif self.state == RobotState.DONE:
            self.handle_done(elapsed_time)
        elif self.state == RobotState.PAUSE:
            self.handle_pause(elapsed_time)

    # ==========================================
    #            STATE HANDLERS
    # ==========================================
    def handle_startup(self, elapsed_time):
        if not self.initial_pose_recorded:
            # Open the gripper at startup:
            self.set_gripper('open')
            
            # Define the scanning pose
            self.scan_x = self.sweep_radius_m * math.cos(self.angle_start_rad)
            self.scan_y = self.sweep_radius_m * math.sin(self.angle_start_rad)
            self.target_roll = math.pi
            self.target_pitch = 0.0
            self.target_yaw = 0.0
            self.initial_pose_recorded = True
        
        self.publish_target(
            self.scan_x, self.scan_y, self.scan_z, 
            roll=self.target_roll, pitch=self.target_pitch, yaw=self.target_yaw
        )
        
        # Calculate where the end-effector is right now
        actual_pos = self.get_current_tcp()
        error = self.calculate_tcp_error([self.scan_x, self.scan_y, self.scan_z], actual_pos)
        
        # Switch to SCANNING when scanning pose is reached
        if error <= self.target_tolerance_m:
            self.reset_vision_flags()
            self.switch_state(RobotState.SCAN_SWEEP)

    def handle_scan_sweep(self, elapsed_time):
        if not self.start_scanning:
            if self.trigger_vision(self.TRIGGER_SWEEP):
                self.get_logger().info("Starting continuous sweep. Triggering camera stream...")

        # Drive the smooth continuous arc sweep
        progress = (1 - math.cos(2 * math.pi * elapsed_time / self.sweep_period_s)) / 2.0
        current_angle = self.interpolate_value(self.angle_start_rad, self.angle_end_rad, progress)
        
        sweep_x = self.sweep_radius_m * math.cos(current_angle)
        sweep_y = self.sweep_radius_m * math.sin(current_angle)
        self.publish_target(sweep_x, sweep_y, self.scan_z)

        # Watch for a quick detection to interrupt the sweep
        if self.latest_cam_weed_pos is not None:
            self.get_logger().info("Possible weed spotted! Stopping arm to confirm...")
            # Lock the exact coordinate we are currently at so the arm freezes
            self.freeze_x = sweep_x
            self.freeze_y = sweep_y
            
            self.reset_vision_flags()
            self.switch_state(RobotState.SCAN_CONFIRM)

    def handle_scan_confirm(self, elapsed_time):
        self.publish_target(self.freeze_x, self.freeze_y, self.scan_z)

        if not self.start_scanning:
            if self.trigger_vision(self.TRIGGER_CONFIRM):
                self.get_logger().info("Sending confirmation trigger...")

        if self.latest_cam_weed_pos is not None:
            self.get_logger().info("Weed confirmed")
            self.switch_state(RobotState.ALIGNING_XY)
            return

        # False Positive Timeout
        if elapsed_time > 10.0:
            self.get_logger().info("False positive. No weed confirmed. Resuming sweep...")
            self.reset_vision_flags()
            self.switch_state(RobotState.SCAN_SWEEP)

    def handle_aligning_xy(self, elapsed_time):
        # Phase 1: The "LOOK"
        if self.locked_weed_base_pos is None:
            if self.latest_cam_weed_pos is None:
                self.get_logger().warn("Waiting for camera data before moving...", throttle_duration_sec=2.0)
                return
            
            # Transform position from camera frame to base frame
            cam_x, cam_y, cam_z = self.latest_cam_weed_pos
            self.locked_weed_base_pos = transform_camera_to_base(
                cam_x, cam_y, cam_z, self.current_joints, 
                imu_roll=self.camera_roll, imu_pitch=self.camera_pitch
            )
            # Save position to log first scan to CSV
            self.first_scan_pos = self.locked_weed_base_pos

            # Calculate where the end-effector is right now (the start of the move)
            self.approach_start_x, self.approach_start_y, self.approach_start_z = self.get_current_tcp()
            
            self.get_logger().info(f"Target locked at Base Pos: [{self.locked_weed_base_pos[0]:.3f}, {self.locked_weed_base_pos[1]:.3f}, {self.locked_weed_base_pos[2]:.3f}]")

        # Phase 2: The "MOVE"
        target_x = self.locked_weed_base_pos[0]
        target_y = self.locked_weed_base_pos[1] + CAMERA_OFFSET_Y
        target_z = self.approach_start_z
        
        distance = self.calculate_tcp_error([target_x, target_y, target_z], [self.approach_start_x, self.approach_start_y, self.approach_start_z])
        move_duration = max(0.2, distance / self.tcp_speed_m_s)
        progress = elapsed_time / move_duration
        
        if progress <= 1.0:
            # Cartesian Path Planner: Interpolate smoothly to the target
            current_x = self.interpolate_value(self.approach_start_x, target_x, progress)
            current_y = self.interpolate_value(self.approach_start_y, target_y, progress)
            self.publish_target(current_x, current_y, self.approach_start_z)
        else:
            self.publish_target(target_x, target_y, target_z)
            
            actual_pos = self.get_current_tcp()
            error = self.calculate_tcp_error([target_x, target_y, target_z], actual_pos)
            
            # Switch to the second scan when the physical arm has reached the correct position
            if error <= self.target_tolerance_m:
                self.reset_vision_flags()
                self.switch_state(RobotState.FINAL_SCAN)

    def handle_final_scan(self, elapsed_time):
        if not self.start_scanning:
            if self.trigger_vision(self.TRIGGER_CONFIRM):
                self.get_logger().info("Sent trigger for Second Scan.")
        
        if self.latest_cam_weed_pos is not None:
            self.switch_state(RobotState.APPROACHING)

    def handle_approaching(self, elapsed_time):
        # Phase 1: The "LOOK"
        if self.locked_weed_base_pos is None:
            if self.latest_cam_weed_pos is None:
                self.get_logger().warn("Waiting for camera data before moving...", throttle_duration_sec=2.0)
                return
            
            # Transform position from camera frame to base frame
            cam_x, cam_y, cam_z = self.latest_cam_weed_pos
            self.locked_weed_base_pos = transform_camera_to_base(
                cam_x, cam_y, cam_z, self.current_joints, 
                imu_roll=self.camera_roll, imu_pitch=self.camera_pitch
            )
            # Save position to log to CSV file
            self.second_scan_pos = self.locked_weed_base_pos

            # Calculate where the end-effector is right now (the start of the move)
            self.approach_start_x, self.approach_start_y, self.approach_start_z = self.get_current_tcp()
                
        # Phase 2: The "MOVE"
        target_x = self.locked_weed_base_pos[0]
        target_y = self.locked_weed_base_pos[1] 
        target_z = self.locked_weed_base_pos[2] + self.hover_offset_z_m
        
        distance = self.calculate_tcp_error([target_x, target_y, target_z], [self.approach_start_x, self.approach_start_y, self.approach_start_z])
        move_duration = max(0.2, distance / self.tcp_speed_m_s)
        progress = elapsed_time / move_duration
        
        if progress <= 1.0:
            # Cartesian Path Planner: Interpolate smoothly to the target
            current_x = self.interpolate_value(self.approach_start_x, target_x, progress)
            current_y = self.interpolate_value(self.approach_start_y, target_y, progress)
            current_z = self.interpolate_value(self.approach_start_z, target_z, progress)
            self.publish_target(current_x, current_y, current_z)
        else:
            self.publish_target(target_x, target_y, target_z)
            self.weeding_start_z = target_z
            self.switch_state(RobotState.PLUNGING)

    def handle_plunging(self, elapsed_time):
        self.target_x = self.locked_weed_base_pos[0]
        self.target_y = self.locked_weed_base_pos[1]
        self.target_z = self.locked_weed_base_pos[2] - self.plunge_depth_m 
        
        distance = abs(self.target_z - self.weeding_start_z)
        move_duration = max(0.2, distance / self.tcp_speed_m_s)
        progress = elapsed_time / move_duration
        
        if progress <= 1.0:
            # Cartesian path planner: Interpolate the Z axis to create a perfectly straight vertical line
            current_z = self.interpolate_value(self.weeding_start_z, self.target_z, progress)
            self.publish_target(self.target_x, self.target_y, current_z)
        else:
            self.publish_target(self.target_x, self.target_y, self.target_z)
        
        # Check if we have physically reached the plunged position
        actual_x, actual_y, actual_z = self.get_current_tcp()
        error = self.calculate_tcp_error([self.target_x, self.target_y, self.target_z], [actual_x, actual_y, actual_z])

        if error <= self.target_tolerance_m:
            self.set_gripper('close')
            self.switch_state(RobotState.GRASPING)

    def handle_grasping(self, elapsed_time):
        if elapsed_time > 2.0: # Waiting for the gripper to physically close
            self.switch_state(RobotState.EXTRACTING)

    def handle_extracting(self, elapsed_time):
        distance = abs(self.target_z - self.weeding_start_z)
        move_duration = max(0.2, distance / self.tcp_speed_m_s)
        progress = elapsed_time / move_duration

        if progress <= 1.0:
            # Cartesian path planner: Interpolate the Z axis to create a perfectly straight vertical line
            current_z = self.interpolate_value(self.target_z, self.weeding_start_z, progress)
            self.publish_target(self.target_x, self.target_y, current_z)
        else:
            self.publish_target(self.target_x, self.target_y, self.weeding_start_z)
            self.switch_state(RobotState.DONE) 

    def handle_done(self, elapsed_time):
        distance = self.calculate_tcp_error([self.target_x, self.target_y, self.weeding_start_z], [self.scan_x, self.scan_y, self.scan_z])
        move_duration = max(0.2, distance / self.tcp_speed_m_s)
        progress = elapsed_time / move_duration

        if progress <= 1.0:
            current_x = self.interpolate_value(self.target_x, self.scan_x, progress)
            current_y = self.interpolate_value(self.target_y, self.scan_y, progress)
            current_z = self.interpolate_value(self.weeding_start_z, self.scan_z, progress)
            self.publish_target(current_x, current_y, current_z)
        else:
            self.publish_target(self.scan_x, self.scan_y, self.scan_z)
            self.get_logger().info("Task complete. Waiting in safe position.", throttle_duration_sec=2.0)
            self.set_gripper('open')
            self.reset_vision_flags()
            self.prompted_for_input = False
            self.switch_state(RobotState.PAUSE)

    def handle_pause(self, elapsed_time):
        if not self.prompted_for_input:
            print("\n" + "="*50)
            print("   JOB DONE! Was the weeding successful?")
            print("   Press [1] for SUCCESS or [0] for FAILED, then press Enter.")
            print("   (System will auto-advance as Success after 20s)")
            print("="*50 + "\n")
            self.prompted_for_input = True
            self.latest_status = "Success" 

        # Non-blocking terminal input check
        i, o, e = select.select([sys.stdin], [], [], 0.0)
        if i:
            user_input = sys.stdin.readline().strip()
            if user_input == '1':
                self.latest_status = "Success"
                self.get_logger().info("User input recorded: SUCCESS. Advancing state immediately...")
            elif user_input == '0':
                self.latest_status = "Failed"
                self.get_logger().info("User input recorded: FAILED. Advancing state immediately...")
            else:
                self.get_logger().warn(f"Unknown input '{user_input}'. Defaulting to Success and advancing...")
            
            self._log_final_status() # Log to CSV right away
            self.switch_state(RobotState.STARTUP)
            return

        if elapsed_time > 20000.0: 
            self.get_logger().info("Pause timeout reached. Automatically logging as Success...")
            self._log_final_status()
            self.switch_state(RobotState.STARTUP)

    def _log_final_status(self):
        """Helper to log the final operation result to CSV."""
        if hasattr(self, 'first_scan_pos'):
            self.log_weed_to_csv(self.first_scan_pos, "First Scan (Aligning)", self.latest_status)
        if hasattr(self, 'second_scan_pos'):
            self.log_weed_to_csv(self.second_scan_pos, "Second Scan (Final Approach)", self.latest_status)


def main(args=None):
    rclpy.init(args=args)
    node = PBVSTaskController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'csv_file') and not node.csv_file.closed:
            node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()