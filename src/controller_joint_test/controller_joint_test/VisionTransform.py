import numpy as np
from .ForwardKinematics import forward_kinematics

def transform_camera_to_base(cam_x, cam_y, cam_z, current_joints):
    """
    Transforms a 3D point (e.g., a weed) from the camera's local coordinate system 
    into the robot's global base coordinate system.
    
    Parameters:
    cam_x, cam_y, cam_z : float : The coordinates of the object seen by the camera (in meters)
    current_joints : list : The 6 joint angles of the robot at the moment the picture was taken
    
    Returns:
    list : [X_base, Y_base, Z_base] representing the physical location on the ground
    """
    
    # 1. Format the target position as a 4x1 homogeneous vector
    P_cam = np.array([cam_x, cam_y, cam_z, 1.0])
    
    # 2. Get the Transformation Matrix of the robot arm (Base to End-Effector)
    # This tells us exactly where the wrist is located right now
    T_base_tool = forward_kinematics(current_joints, verbose=False)
    
    # 3. Define the physical offset of the camera lens relative to the tool flange
    offset_x = 0.05  # Distance forward/backward (meters)
    offset_y = 0.00  # Distance left/right (meters)
    offset_z = 0.03  # Distance up/down (meters)
    
    # The camera is facing the exact same direction as the tool flange so no rotation is required
    # We build the static transformation matrix for the camera.
    T_tool_cam = np.array([
        [1.0, 0.0, 0.0, offset_x],
        [0.0, 1.0, 0.0, offset_y],
        [0.0, 0.0, 1.0, offset_z],
        [0.0, 0.0, 0.0, 1.0 ]
    ])
    
    # 4. Multiply the matrices to get the camera's position relative to the base
    T_base_cam = np.dot(T_base_tool, T_tool_cam)
    
    # 5. Transform the weed's coordinate into the base frame
    P_base = np.dot(T_base_cam, P_cam)
    
    # Return just the X, Y, Z coordinates
    return [P_base[0], P_base[1], P_base[2]]

# --- TESTING ---
if __name__ == "__main__":
    # Test scenario: The robot is pointing straight down at the table
    test_joints = [0.0, np.pi/4, np.pi/4, 0.0, -np.pi/2, 0.0]
    
    # Simulated output from YOLO + Depth sensor (The weed is 40 cm straight ahead from the lens)
    weed_cam_x = 0.0
    weed_cam_y = 0.0
    weed_cam_z = 0.40 
    
    print("--- Vision Transformation Test ---")
    print(f"Weed detected at camera coordinates: X={weed_cam_x}, Y={weed_cam_y}, Z={weed_cam_z}")
    
    try:
        # Calculate the real-world coordinates
        real_world_pos = transform_camera_to_base(weed_cam_x, weed_cam_y, weed_cam_z, test_joints)
        print(f"Resulting BASE coordinates for IK: X={real_world_pos[0]:.3f}, Y={real_world_pos[1]:.3f}, Z={real_world_pos[2]:.3f}")
    except Exception as e:
        print(f"Transformation failed. Did you import ForwardKinematics correctly? Error: {e}")