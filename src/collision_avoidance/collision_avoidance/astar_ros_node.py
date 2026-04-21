import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
from visualization_msgs.msg import Marker

# --- PURE PYTHON IMPORTS ---
from .CSpaceGrid import CSpaceGrid
from .AstarSearch import AStarPlanner
from .StringPuller import StringPuller
from .TrajectoryExecutor import LinearTrajectoryExecutor

# Import Forward Kinematics 
from controller_joint_test.ForwardKinematics import forward_kinematics

class MasterAStarNode(Node):
    def __init__(self):
        super().__init__('master_astar_node')
        
        # 1. Setup the Map
        self.joint_limits = [  
            [-3.12, 3.12], 
            [-1.05, 1.95], 
            [-2.27, 2.27], 
            [-2.09, 2.09], 
            [-2.0, 2.0]
        ]

        # Adjust these so the markers align with your RViz robot model
        self.viz_offset_x = 0.148
        self.viz_offset_y = 0.15
        self.viz_offset_z = 0.0  
        
        # --- THE SINGLE SOURCE OF TRUTH ---
        # Define everything in MATH coordinates (relative to your DH origin)
        self.obstacles = [
            {
                'id': 0,
                'type': 'cylinder',
                'name': 'tree_trunk',
                'x': 0.4, 'y': 0.0,
                'radius': 0.15,
                'height': 1.0,
                'color': [0.5, 0.3, 0.0, 0.8] # Brown
            },
            # {
            #     'id': 1,
            #     'type': 'box',
            #     'name': 'side_wall',
            #     'x_min': -0.3, 'x_max': 0.7,
            #     'y_min': 0.35, 'y_max': 0.45,
            #     'z_min': 0.0,  'z_max': 1.0,
            #     'color': [0.6, 0.6, 0.6, 0.8] # Gray
            # }
        ]

        # Robot joint radius and safety buffer to objects
        self.robot_link_radius = 0.05
        self.safety_buffer = 0.03

        self.grid = CSpaceGrid(self.joint_limits, k=10)
        
        self.planner = AStarPlanner(self.grid, self.check_collision)
        self.smoother = StringPuller(self.check_collision)
        
        self.trajectory_executor = LinearTrajectoryExecutor(publish_rate_hz=50.0)
        
        # State variables
        self.current_joints = None
        self.is_planning = False

        # ROS Infrastructure
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.goal_sub = self.create_subscription(Float64MultiArray, '/joint_goal_test', self.goal_callback, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)

        # RViz Marker Publisher
        self.marker_pub = self.create_publisher(Marker, '/obstacle_marker', 10)
        # Publish the tree once every second so RViz never loses it
        self.marker_timer = self.create_timer(1.0, self.publish_obstacle_marker)
        
        # The single ROS Timer that drives the executor!
        self.timer = self.create_timer(1.0 / 50.0, self.timer_callback)
            
        self.get_logger().info("Master A* Node Ready! Waiting for goals...")

    # --- THE COLLISION CHECKER (Wrapper for your FK) ---
    def check_collision(self, test_joints):
        # Safely pad the array with a zero for the 6-DOF FK math
        padded_joints = test_joints.tolist() + [0.0]
        
        _, skeleton_points = forward_kinematics(padded_joints, return_skeleton=True)
        
        total_padding = self.robot_link_radius + self.safety_buffer

        # 1. Generate all 3D checkpoints along the solid arm FIRST
        checkpoints = []
        for i in range(len(skeleton_points) - 1):
            # Grab full X, Y, Z coordinates
            p1 = np.array(skeleton_points[i])
            p2 = np.array(skeleton_points[i+1])
            
            # Create 10 checkpoints along this specific link
            for t in np.linspace(0.0, 1.0, 10):
                checkpoints.append(p1 + t * (p2 - p1))

        # 2. Test every checkpoint against every obstacle
        for pt in checkpoints:
            px, py, pz = pt[0], pt[1], pt[2]
            
            for obs in self.obstacles:
                
                # --- CYLINDER MATH (Ignores Z-axis) ---
                if obs['type'] == 'cylinder':
                    dist = np.sqrt((px - obs['x'])**2 + (py - obs['y'])**2)
                    if dist <= (obs['radius'] + total_padding):
                        return False 
                        
                # --- SPHERE MATH (Checks X, Y, Z) ---
                elif obs['type'] == 'sphere':
                    dist = np.sqrt((px - obs['x'])**2 + (py - obs['y'])**2 + (pz - obs['z'])**2)
                    if dist <= (obs['radius'] + total_padding):
                        return False
                        
                # --- BOX/AABB MATH (Checks boundaries) ---
                elif obs['type'] == 'box':
                    # We add the total_padding directly to the box walls!
                    if (obs['x_min'] - total_padding <= px <= obs['x_max'] + total_padding) and \
                       (obs['y_min'] - total_padding <= py <= obs['y_max'] + total_padding) and \
                       (obs['z_min'] - total_padding <= pz <= obs['z_max'] + total_padding):
                        return False

        return True

    # --- ROS CALLBACKS ---
    def joint_state_callback(self, msg):
        expected_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        if not all(joint in msg.name for joint in expected_joints): return
        
        ordered = [msg.position[msg.name.index(j)] for j in expected_joints]
        self.current_joints = np.array(ordered)

    def goal_callback(self, msg):
        if self.current_joints is None or self.is_planning or self.trajectory_executor.is_moving:
            return

        self.is_planning = True
        target_joints = np.array(msg.data[:5])

        # ==========================================
        # NEW DIAGNOSTIC PRINTOUT
        # ==========================================

        self.current_joints_6DOF_dummy = np.append(self.current_joints, [0.0])
        target_joints_6DOF_dummy = np.append(target_joints, [0.0])

        _, start_skeleton = forward_kinematics(self.current_joints_6DOF_dummy + [0], return_skeleton=True)
        _, goal_skeleton = forward_kinematics(target_joints_6DOF_dummy + [0], return_skeleton=True)
        
        start_tip = start_skeleton[-1]
        goal_tip = goal_skeleton[-1]
        
        self.get_logger().warn(f"--- KINEMATICS REALITY CHECK ---")
        self.get_logger().warn(f"START TIP IS AT -> X: {start_tip[0]:.3f}, Y: {start_tip[1]:.3f}, Z: {start_tip[2]:.3f}")
        self.get_logger().warn(f"GOAL TIP IS AT  -> X: {goal_tip[0]:.3f}, Y: {goal_tip[1]:.3f}, Z: {goal_tip[2]:.3f}")
        
        self.get_logger().info("Planning...")

        planner_time = 5
        raw_path = self.planner.plan(self.current_joints, target_joints, planner_time)
        
        if raw_path:
            self.get_logger().info(f"Path found! Smoothing...")
            smooth_path = self.smoother.smooth(raw_path)
            
            self.get_logger().info(f"Executing {len(smooth_path)} waypoints...")
            self.trajectory_executor.load_path(smooth_path)

        else:
            self.get_logger().error("Planning failed")
            
        self.is_planning = False

    def timer_callback(self):
        """ The 50Hz control loop """
        next_pose = self.trajectory_executor.step()
        if next_pose is not None:
            msg = Float64MultiArray()
            msg.data = next_pose
            self.cmd_pub.publish(msg)

    
    def publish_obstacle_marker(self):
        """ Object publisher """
        for obs in self.obstacles:
            marker = Marker()
            marker.header.frame_id = "link_0"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "environment"
            marker.id = obs['id']
            marker.action = Marker.ADD
            
            # Set the color from our definition
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = obs['color']

            if obs['type'] == 'cylinder':
                marker.type = Marker.CYLINDER
                # Apply Offset to Position
                marker.pose.position.x = obs['x'] + self.viz_offset_x
                marker.pose.position.y = obs['y'] + self.viz_offset_y
                marker.pose.position.z = (obs['height'] / 2.0) + self.viz_offset_z
                
                # Diameter is 2 * radius
                marker.scale.x = obs['radius'] * 2.0
                marker.scale.y = obs['radius'] * 2.0
                marker.scale.z = obs['height']

            elif obs['type'] == 'box':
                marker.type = Marker.CUBE
                # Calculate Center
                cx = (obs['x_min'] + obs['x_max']) / 2.0
                cy = (obs['y_min'] + obs['y_max']) / 2.0
                cz = (obs['z_min'] + obs['z_max']) / 2.0
                
                marker.pose.position.x = cx + self.viz_offset_x
                marker.pose.position.y = cy + self.viz_offset_y
                marker.pose.position.z = cz + self.viz_offset_z
                
                # Calculate Scale (Size)
                marker.scale.x = obs['x_max'] - obs['x_min']
                marker.scale.y = obs['y_max'] - obs['y_min']
                marker.scale.z = obs['z_max'] - obs['z_min']

            marker.pose.orientation.w = 1.0
            self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = MasterAStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()