import numpy as np
from math import cos, sin
from .ForwardKinematics import forward_kinematics, rotation_matrix_to_fixed_angles
from .configuration import CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z

def euler_to_rot_matrix(roll, pitch, yaw):
    """Constructs a rotation matrix from Roll (X), Pitch (Y), Yaw (Z) in radians"""
    R_x = np.array([[1, 0, 0],
                    [0, cos(roll), -sin(roll)],
                    [0, sin(roll),  cos(roll)]])
    
    R_y = np.array([[cos(pitch), 0, sin(pitch)],
                    [0, 1, 0],
                    [-sin(pitch), 0, cos(pitch)]])
    
    R_z = np.array([[cos(yaw), -sin(yaw), 0],
                    [sin(yaw),  cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def transform_camera_to_base(cam_x, cam_y, cam_z, current_joints, imu_roll=None, imu_pitch=None):
    """
    Transforms a 3D point (e.g., a weed) from the camera's local coordinate system 
    into the robot's global base coordinate system.
    
    Parameters:
    cam_x, cam_y, cam_z : float : The coordinates of the object seen by the camera (in meters)
    current_joints : list : The 6 joint angles of the robot at the moment the picture was taken
    If imu_roll and imu_pitch are provided (in radians), they override the theoretical
    Forward Kinematics orientation to compensate for physical backlash.

    Returns:
    list : [X_base, Y_base, Z_base] representing the physical location on the ground
    """
    # Format the target position as a 4x1 homogeneous vector
    P_cam = np.array([cam_x, cam_y, cam_z, 1.0])
    
    # Get where the robot thinks it is from the encoders
    T_base_tool = forward_kinematics(current_joints, verbose=False)
    
    # Add the static camera offset
    T_tool_cam = np.array([
        [1.0, 0.0, 0.0, CAMERA_OFFSET_X],
        [0.0, 1.0, 0.0, CAMERA_OFFSET_Y],
        [0.0, 0.0, 1.0, CAMERA_OFFSET_Z],
        [0.0, 0.0, 0.0, 1.0 ]
    ])

    T_base_cam = np.dot(T_base_tool, T_tool_cam)

    # Apply IMU correction if data is available
    if imu_roll is not None and imu_pitch is not None:
        # Determine the theoretical yaw from FK
        theoretical_yaw = np.arctan2(T_base_cam[1, 0], T_base_cam[0, 0])
        
        # New IMU-corrected rotation matrix
        R_corrected_absolute = euler_to_rot_matrix(imu_roll, imu_pitch, theoretical_yaw)
        
        # Overwrite the theoretical orientation with the real-world measured orientation
        T_base_cam[:3, :3] = R_corrected_absolute

    # Transform the weed coordinate using the corrected matrix
    P_base = np.dot(T_base_cam, P_cam)
    
    return [P_base[0], P_base[1], P_base[2]]

# --- TESTING ---
if __name__ == "__main__":
    # Test scenario: 
    test_joints = [-0.786, 0.281, -0.990, 0.0, -1.870, 0.0]
    
    # Simulated output from YOLO + Depth sensor (The weed is 40 cm straight ahead from the lens)
    weed_cam_x = 0.0
    weed_cam_y = 0.0
    weed_cam_z = 0.40 
    
    print("--- Vision Transformation Test ---")
    print(f"Weed detected at camera coordinates: X={weed_cam_x}, Y={weed_cam_y}, Z={weed_cam_z}")
    
    # Calculate the real-world coordinates
    real_world_pos = transform_camera_to_base(weed_cam_x, weed_cam_y, weed_cam_z, test_joints)