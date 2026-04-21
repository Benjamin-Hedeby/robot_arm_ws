import numpy as np
from math import cos, sin, pi

def get_modified_dh_matrix(alpha, a, d, theta):
    """
    Calculates the transformation matrix using Craig's Modified DH convention.

    Parameters:
    alpha (float): Twist angle (alpha_{i-1})
    a (float): Link length (a_{i-1})
    d (float): Link offset (d_i)
    theta (float): Joint angle (theta_i)

    Returns:
    numpy.ndarray: 4x4 Homogeneous Transformation Matrix
    """
    c_th = cos(theta)
    s_th = sin(theta)
    c_alp = cos(alpha)
    s_alp = sin(alpha)

    # Formula from Craig's "Introduction to Robotics", Eq. 3.6
    T = np.array([
        [c_th,          -s_th,           0,           a],
        [s_th * c_alp,   c_th * c_alp,  -s_alp,      -d * s_alp],
        [s_th * s_alp,   c_th * s_alp,   c_alp,       d * c_alp],
        [0,              0,              0,           1]
    ])
    
    return T

def forward_kinematics(joints, return_skeleton=False, verbose=False):
    """
    Computes the End-Effector position and orientation based on joint angles.

    Parameters:
    joints (list): List of 6 joint angles in radians [q1, q2, q3, q4, q5, q6]
    verbose (bool): If True, prints intermediate frame positions for debugging.

    Returns:
    numpy.ndarray: Final 4x4 transformation matrix (Base to End-Effector)
    """
    # 1. Define the DH Parameters based on the physical robot
    # Format: [alpha_(i-1), a_(i-1), d_i]
    # Note: Units are in METERS
    dh_table = [
        [0,       0,      0],        # Frame 1
        [pi / 2,  0.040,  0],        # Frame 2
        [0,       0.300,  0],        # Frame 3
        [pi / 2,  0.000,  0.380],    # Frame 4
        [-pi / 2, 0,      0],        # Frame 5
        [pi / 2,  0,      0.201]     # Frame 6 (End-Effector)
    ]

    # Initialize the total transformation as an Identity Matrix (4x4)
    T_total = np.eye(4)
    skeleton_points = []

    if verbose:
        print("-" * 30)
        print("Computing Forward Kinematics:")

    # Loop through each joint and multiply matrices sequentially
    for i, params in enumerate(dh_table):
        alpha = params[0]
        a = params[1]
        d = params[2]
        theta = joints[i]

        # Convert from physical motor angle q to mathematical DH angle theta
        if i == 0:
            theta = -theta
        elif i == 1:
            theta = -theta + pi/2
        elif i == 2:
            theta = theta + pi/2
           # Calculate transformation for current link
        T_i = get_modified_dh_matrix(alpha, a, d, theta)

        # Multiply to the chain: T_total = T_total * T_i
        T_total = np.dot(T_total, T_i)

        joint_position = T_total[:3, 3]
        skeleton_points.append(joint_position)

        # Print position of each joint frame if debugging is enabled
        if verbose:
            pos = T_total[:3, 3]
            print(f"Frame {i + 1} pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    if return_skeleton:
        return T_total, skeleton_points
    else:
        return T_total


def rotation_matrix_to_fixed_angles(T):
    """
    Converts a 4x4 Transformation Matrix to Roll, Pitch, Yaw (RPY) fixed angles.
    
    Returns:
    numpy.ndarray: Array of angles [gamma, beta, alpha] in degrees.
    """
    # Calculate sy (sine of y) to check for singularity (Gimbal Lock)
    sy = np.sqrt(T[0, 0]**2 + T[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        # Standard analytical solution
        gamma = np.arctan2(T[2, 1], T[2, 2])
        beta = np.arctan2(-T[2, 0], sy)
        alpha = np.arctan2(T[1, 0], T[0, 0])
    else:
        # Singularity handling: The robot is pointing straight up or down
        gamma = np.arctan2(-T[1, 2], T[1, 1])
        beta = np.arctan2(-T[2, 0], sy)
        alpha = 0

    # Convert radians to degrees for readability
    return np.degrees([gamma, beta, alpha])

# --- TESTING ---
if __name__ == "__main__":
    
    # Test 1: Given position
    test_joints = [0, np.pi / 2, np.pi / 2, 0, 0, 0]

    # Run the kinematics with verbose enabled to see intermediate steps
    final_transform = forward_kinematics(test_joints, verbose=True)

    print("-" * 30)
    print("Final End-Effector Matrix:")
    print(np.round(final_transform, 4))

    # Extract X, Y, Z
    x, y, z = final_transform[:3, 3]

    # Extract Orientation
    rpy = rotation_matrix_to_fixed_angles(final_transform)
    
    print("-" * 30)
    print(f"Position (XYZ): [{x:.3f}, {y:.3f}, {z:.3f}] meters")
    print(f"Orientation (RPY): [Gamma/Roll: {rpy[0]:.1f}°, Beta/Pitch: {rpy[1]:.1f}°, Alpha/Yaw: {rpy[2]:.1f}°]")