import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time

class SquareTestNode(Node):
    def __init__(self):
        super().__init__('square_test_node')
        
        # 1. PUBLISHER: Send coordinates to IK Streamer
        self.target_pub = self.create_publisher(Float64MultiArray, '/desired_tcp_pose_euler', 10)
        
        # --- CONFIGURATION ---
        self.HOVER_Z = -0.45      # Hover over the table
        self.PAPER_Z = -0.505     # Exact height of the paper 
        
        self.duration_per_line = 3.0  # 3 seconds to draw each line
        
        # Define the 4 corners of a 10x10 cm square
        # Centered at X=0.55, Y=0.0
        self.corners = [
            (0.50, -0.05), # Corner 0: Bottom Right
            (0.50, -0.00), # Corner 1: Top Right
            (0.50,  0.05), # Corner 2: Top Left
            (0.50,  0.10), # Corner 3: Bottom Left
            (0.50,  0.15)  # Back to Corner 0 to close the square
        ]
        
        # State machine variables
        self.current_waypoint = -1 # -1 means hovering, 0-4 is drawing, 5 is lifting
        self.state_start_time = time.time()
        
        # Start hovering over the first corner
        self.start_pos = (self.corners[0][0], self.corners[0][1], self.HOVER_Z)
        self.target_pos = (self.corners[0][0], self.corners[0][1], self.PAPER_Z)
        
        # Run loop at 100 Hz for smooth drawing
        self.timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info("Square Test Node Started! Hovering over starting point...")

    def control_loop(self):
        current_time = time.time()
        elapsed_time = current_time - self.state_start_time
        
        # Calculate progress
        progress = elapsed_time / self.duration_per_line
        if progress > 1.0:
            progress = 1.0
            
        # Linear Interpolation for XYZ
        current_x = self.start_pos[0] + (self.target_pos[0] - self.start_pos[0]) * progress
        current_y = self.start_pos[1] + (self.target_pos[1] - self.start_pos[1]) * progress
        current_z = self.start_pos[2] + (self.target_pos[2] - self.start_pos[2]) * progress
        
        # Always point straight down
        roll = math.pi
        pitch = 0.0
        yaw = 0.0
        
        self.publish_target(current_x, current_y, current_z, roll, pitch, yaw)
        
        # Check if we reached the target
        if progress == 1.0:
            self.advance_state()

    def advance_state(self):
        self.state_start_time = time.time()
        
        if self.current_waypoint == -1:
            # We finished dropping the pen to the paper. Start drawing to corner 1!
            self.get_logger().info("Pen on paper. Drawing line 1...")
            self.current_waypoint = 1
            self.start_pos = self.target_pos
            self.target_pos = (self.corners[1][0], self.corners[1][1], self.PAPER_Z)
            
        elif self.current_waypoint >= 1 and self.current_waypoint < 4:
            # Drawing lines 2, 3, and 4
            self.current_waypoint += 1
            self.get_logger().info(f"Drawing line {self.current_waypoint}...")
            self.start_pos = self.target_pos
            self.target_pos = (self.corners[self.current_waypoint][0], self.corners[self.current_waypoint][1], self.PAPER_Z)
            
        elif self.current_waypoint == 4:
            # Square is complete! Lift the pen
            self.get_logger().info("Square complete! Lifting pen...")
            self.current_waypoint = 5
            self.start_pos = self.target_pos
            self.target_pos = (self.start_pos[0], self.start_pos[1], self.HOVER_Z)
            
        elif self.current_waypoint == 5:
            # We are done
            self.get_logger().info("Test finished. Shutting down timer.")
            self.timer.cancel()

    def publish_target(self, x, y, z, roll, pitch, yaw):
        msg = Float64MultiArray()
        msg.data = [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]
        self.target_pub.publish(msg)

class SquareDotTestNode(Node):
    def __init__(self):
        super().__init__('square_test_node')
        
        # 1. PUBLISHER: Send coordinates to IK Streamer
        self.target_pub = self.create_publisher(Float64MultiArray, '/desired_tcp_pose_euler', 10)
        
        # --- CONFIGURATION ---
        self.HOVER_Z = -0.45      # Hover over the table
        self.PAPER_Z = -0.509     # Exact height of the paper 
        
        # Durations for the different movements to ensure smooth motion
        self.duration_air_move = 2  # Seconds to move horizontally between corners
        self.duration_poke = 2      # Seconds to move down to the paper / up from the paper
        self.duration_hold = 0.5      # Seconds to hold the pen on the paper to leave a mark
        
        # Define the 4 corners of a 10x10 cm square
        self.corners = [
            (0.50, -0.05), # Corner 0: Bottom Right
            (0.60, -0.05), # Corner 1: Top Right
            (0.60,  0.05), # Corner 2: Top Left
            (0.50,  0.05)  # Corner 3: Bottom Left
        ]
        
        # --- STATE MACHINE VARIABLES ---
        self.max_loops = 10
        self.current_loop = 1
        self.current_corner_idx = 0
        
        # Define the phases of the poking motion
        self.PHASE_MOVE_AIR = 0
        self.PHASE_POKE_DOWN = 1
        self.PHASE_HOLD = 2
        self.PHASE_LIFT_UP = 3
        
        self.current_phase = self.PHASE_MOVE_AIR
        self.current_duration = self.duration_air_move
        
        # Initial starting position (Hovering over corner 0)
        start_x, start_y = self.corners[0]
        self.start_pos = (start_x, start_y, self.HOVER_Z)
        self.target_pos = (start_x, start_y, self.HOVER_Z)
        
        self.state_start_time = time.time()
        
        # Run loop at 100 Hz for smooth drawing
        self.timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info(f"Dot Test Started! Loop 1 of {self.max_loops}...")

    def control_loop(self):
        current_time = time.time()
        elapsed_time = current_time - self.state_start_time
        
        # Calculate progress from 0.0 to 1.0
        progress = elapsed_time / self.current_duration
        if progress > 1.0:
            progress = 1.0
            
        # Linear Interpolation (Lerp) for XYZ
        current_x = self.start_pos[0] + (self.target_pos[0] - self.start_pos[0]) * progress
        current_y = self.start_pos[1] + (self.target_pos[1] - self.start_pos[1]) * progress
        current_z = self.start_pos[2] + (self.target_pos[2] - self.start_pos[2]) * progress
        
        # Keep orientation pointing straight down
        roll = math.pi
        pitch = 0.0
        yaw = 0.0
        
        self.publish_target(current_x, current_y, current_z, roll, pitch, yaw)
        
        # Check if the current movement phase is complete
        if progress == 1.0:
            self.advance_state()

    def advance_state(self):
        self.state_start_time = time.time()
        
        # Get the current X and Y of the active corner
        corner_x, corner_y = self.corners[self.current_corner_idx]
        
        if self.current_phase == self.PHASE_MOVE_AIR:
            # We arrived above the corner. Time to poke down.
            self.current_phase = self.PHASE_POKE_DOWN
            self.start_pos = (corner_x, corner_y, self.HOVER_Z)
            self.target_pos = (corner_x, corner_y, self.PAPER_Z)
            self.current_duration = self.duration_poke
            
        elif self.current_phase == self.PHASE_POKE_DOWN:
            # We reached the paper. Hold it there to make a clear dot.
            self.current_phase = self.PHASE_HOLD
            self.start_pos = (corner_x, corner_y, self.PAPER_Z)
            self.target_pos = (corner_x, corner_y, self.PAPER_Z)
            self.current_duration = self.duration_hold
            
        elif self.current_phase == self.PHASE_HOLD:
            # Holding is done. Lift back up into the air.
            self.current_phase = self.PHASE_LIFT_UP
            self.start_pos = (corner_x, corner_y, self.PAPER_Z)
            self.target_pos = (corner_x, corner_y, self.HOVER_Z)
            self.current_duration = self.duration_poke
            
        elif self.current_phase == self.PHASE_LIFT_UP:
            # We are back in the air. Move to the next corner.
            self.current_corner_idx += 1
            
            # Check if we completed all 4 corners
            if self.current_corner_idx >= 4:
                self.current_corner_idx = 0
                self.current_loop += 1
                
                # Check if we completed all loops
                if self.current_loop > self.max_loops:
                    self.get_logger().info("10 loops completed! Shutting down test.")
                    self.timer.cancel()
                    return
                else:
                    self.get_logger().info(f"Starting loop {self.current_loop} of {self.max_loops}...")
            
            # Set up the movement to the new corner
            next_corner_x, next_corner_y = self.corners[self.current_corner_idx]
            self.current_phase = self.PHASE_MOVE_AIR
            self.start_pos = (corner_x, corner_y, self.HOVER_Z)
            self.target_pos = (next_corner_x, next_corner_y, self.HOVER_Z)
            self.current_duration = self.duration_air_move

    def publish_target(self, x, y, z, roll, pitch, yaw):
        msg = Float64MultiArray()
        msg.data = [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]
        self.target_pub.publish(msg)

class SineWaveTestNode(Node):
    def __init__(self):
        super().__init__('sine_wave_test_node')
        
        # 1. PUBLISHER: Send coordinates to IK Streamer
        self.target_pub = self.create_publisher(Float64MultiArray, '/desired_tcp_pose_euler', 10)
        
        # --- CONFIGURATION ---
        self.HOVER_Z = -0.20      # Hover over the table
        self.PAPER_Z = -0.3037    # Exact height of the paper
        
        self.duration_drawing = 10.0  # Take 10 seconds to draw the wave for smooth motion
        
        # Sine wave parameters
        self.base_x = 0.55        # Center distance from robot base
        self.amplitude = 0.05     # Wave height: +/- 5 cm in X-axis
        self.start_y = 0.10       # Start 15 cm to the left
        self.end_y = -0.10        # End 15 cm to the right
        self.num_waves = 2.0      # Draw 2 full sine waves
        
        # --- STATE MACHINE VARIABLES ---
        self.PHASE_MOVE_AIR = 0
        self.PHASE_POKE_DOWN = 1
        self.PHASE_DRAWING = 2
        self.PHASE_LIFT_UP = 3
        self.PHASE_DONE = 4
        
        self.current_phase = self.PHASE_MOVE_AIR
        self.state_start_time = time.time()
        
        # Run loop at 100 Hz for high-resolution path generation
        self.timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info("Sine Wave Node Started! Moving to start position...")

    def control_loop(self):
        current_time = time.time()
        elapsed_time = current_time - self.state_start_time
        
        # Keep orientation pointing straight down
        roll = math.pi
        pitch = 0.0
        yaw = 0.0
        
        if self.current_phase == self.PHASE_MOVE_AIR:
            # Hover over the starting point for 2 seconds to stabilize
            target_x, target_y = self.calculate_sine_point(0.0)
            self.publish_target(target_x, target_y, self.HOVER_Z, roll, pitch, yaw)
            if elapsed_time > 2.0:
                self.advance_state()
                
        elif self.current_phase == self.PHASE_POKE_DOWN:
            # Drop down to the paper smoothly over 1 second
            progress = min(elapsed_time / 1.0, 1.0)
            target_x, target_y = self.calculate_sine_point(0.0)
            current_z = self.HOVER_Z + (self.PAPER_Z - self.HOVER_Z) * progress
            self.publish_target(target_x, target_y, current_z, roll, pitch, yaw)
            if progress >= 1.0:
                self.advance_state()
                
        elif self.current_phase == self.PHASE_DRAWING:
            # Draw the continuous sine wave based on time progress
            progress = min(elapsed_time / self.duration_drawing, 1.0)
            current_x, current_y = self.calculate_sine_point(progress)
            self.publish_target(current_x, current_y, self.PAPER_Z, roll, pitch, yaw)
            if progress >= 1.0:
                self.advance_state()
                
        elif self.current_phase == self.PHASE_LIFT_UP:
            # Lift the pen straight up from the end point
            progress = min(elapsed_time / 1.0, 1.0)
            target_x, target_y = self.calculate_sine_point(1.0)
            current_z = self.PAPER_Z + (self.HOVER_Z - self.PAPER_Z) * progress
            self.publish_target(target_x, target_y, current_z, roll, pitch, yaw)
            if progress >= 1.0:
                self.advance_state()

    def calculate_sine_point(self, progress):
        # Calculate Y moving linearly from left to right
        current_y = self.start_y + (self.end_y - self.start_y) * progress
        
        # Calculate X moving in a sine wave pattern
        # Uses standard formula: X = Base + Amplitude * sin(t * frequency * 2PI)
        current_x = self.base_x + self.amplitude * math.sin(progress * self.num_waves * 2 * math.pi)
        
        return current_x, current_y

    def advance_state(self):
        self.state_start_time = time.time()
        self.current_phase += 1
        
        if self.current_phase == self.PHASE_POKE_DOWN:
            self.get_logger().info("Dropping pen to paper...")
        elif self.current_phase == self.PHASE_DRAWING:
            self.get_logger().info("Drawing sine wave...")
        elif self.current_phase == self.PHASE_LIFT_UP:
            self.get_logger().info("Done drawing. Lifting pen...")
        elif self.current_phase == self.PHASE_DONE:
            self.get_logger().info("Test finished successfully!")
            self.timer.cancel()

    def publish_target(self, x, y, z, roll, pitch, yaw):
        msg = Float64MultiArray()
        msg.data = [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]
        self.target_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SquareDotTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()