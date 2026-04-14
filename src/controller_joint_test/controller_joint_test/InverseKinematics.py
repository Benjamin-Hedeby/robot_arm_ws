import numpy as np
from math import cos, sin, pi

def inverse_kinematics(position, orientation):

    a_1 = 0.045
    a_2 = 0.300
    d_4 = 0.334
    x_ee, y_ee, z_ee = position         # End-effector position
    gamma, beta, alpha = orientation    # End-effector orientation (X-Y-Z fixed angles)

    # Calculate the rotation matrix of the end-effector R_0_6 (target orientation)
    R_0_6 = fixed_angles_to_rotation_matrix(gamma, beta, alpha)

    # Calculate the Wrist Center (P_wc)
    # d6 is the distance from the wrist center (intersection of Z4, Z5, Z6) to the end-effector.
    d6 = 0.2028

    # The Z-axis of the end-effector in the base frame is the 3rd column of R_0_6
    Z_ee = R_0_6[:, 2]

    # Subtract the gripper length along the Z-axis vector to find the Wrist Center
    x_wc = x_ee - d6 * Z_ee[0]
    y_wc = y_ee - d6 * Z_ee[1]
    z_wc = z_ee - d6 * Z_ee[2]

    # Solve for the arm joint angles (Theta 1, 2, 3) using Wrist Center
    theta1 = np.arctan2(y_wc, x_wc)

    x_1 = np.sqrt(x_wc**2 + y_wc**2)
    R = np.sqrt((x_1 - a_1)**2 + z_wc**2)
    phi1 = np.arctan2(z_wc, x_1-a_1)
    arccos_argument = (R**2+a_2**2-d_4**2)/(2*a_2*R)
    arccos_argument = np.clip(arccos_argument, -1.0, 1.0)
    phi2 = np.arccos(arccos_argument)

    theta2 = phi2 + phi1

    arccos_argument = (a_2 ** 2 + d_4 ** 2 - R ** 2) / (2 * a_2 * d_4)
    arccos_argument = np.clip(arccos_argument, -1.0, 1.0)
    psi = np.arccos(arccos_argument)
    theta3 = psi - np.pi/2

    # R_0_4_zero is the rotation matrix from frame 0 to 4 evaluated at theta_4 = 0
    R_0_4_zero = get_R_0_4_eval_zero(theta1, theta2, theta3)

    # Calculate the wrist rotation matrix R_4_6
    R_wrist = np.dot(R_0_4_zero.T, R_0_6)

    r11, r12, r13 = R_wrist[0, 0], R_wrist[0, 1], R_wrist[0, 2]
    r21, r22, r23 = R_wrist[1, 0], R_wrist[1, 1], R_wrist[1, 2]
    r31, r32, r33 = R_wrist[2, 0], R_wrist[2, 1], R_wrist[2, 2]


    # Calculate theta_5 (Beta)
    # Calculate the positive and negative roots
    # This represents the two physical ways the wrist can achieve the orientation
    pos_root = np.sqrt(r31 ** 2 + r32 ** 2)
    neg_root = -pos_root

    # Check for singularity (Gimbal Lock) where sin(beta) is approx 0
    if abs(pos_root) < 1e-6:
        # We arbitrarily set theta_4 to 0.0 and let theta_6 do the rotation
        theta4 = 0.0

        if r33 > 0:
            # Case: beta is near 0 degrees
            theta5 = 0.0
            theta6 = np.arctan2(-r12, r11)
        else:
            # Case: beta is near 180 degrees (pi radians)
            theta5 = pi
            theta6 = np.arctan2(r12, -r11)
    else:
        # Normal, non-singular case

        # Calculate Solution 1 (Positive theta_5 root)
        theta5_1 = np.arctan2(pos_root, r33)
        theta4_1 = np.arctan2(r23, r13)
        theta6_1 = np.arctan2(r32, -r31)

        # Calculate Solution 2 (Negative theta_5 root)
        theta5_2 = np.arctan2(neg_root, r33)
        theta4_2 = np.arctan2(-r23, -r13)
        theta6_2 = np.arctan2(-r32, r31)

        # Choose the solution that keeps theta_4 and theta_6 closest to zero
        # We do this by comparing the sum of their absolute values
        cost_1 = abs(theta4_1) + abs(theta6_1)
        cost_2 = abs(theta4_2) + abs(theta6_2)

        if cost_1 <= cost_2:
            theta4, theta5, theta6 = theta4_1, theta5_1, theta6_1
        else:
            theta4, theta5, theta6 = theta4_2, theta5_2, theta6_2

    # Mathematical DH angles (theta) for joints 1-6 based on the IK solution
    #joints = [theta1, theta2, theta3, theta4, theta5, theta6]

    # Convert mathematical DH angles theta to physical motor angles (q). Based on the offsets
    joints = [-theta1, -(theta2 - pi/2), theta3 - pi/2, -theta4, theta5, -theta6]

    return joints


def fixed_angles_to_rotation_matrix(gamma, beta, alpha):
    """
    Calculates the 3x3 rotation matrix R_0_6 based on X-Y-Z fixed angles.

    Parameters:
    gamma : float : Rotation around the fixed X-axis (Roll) in radians
    beta  : float : Rotation around the fixed Y-axis (Pitch) in radians
    alpha : float : Rotation around the fixed Z-axis (Yaw) in radians

    Returns:
    numpy.ndarray : A 3x3 rotation matrix
    """

    # Calculate sines and cosines to make the matrix readable
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    cb = np.cos(beta)
    sb = np.sin(beta)

    cg = np.cos(gamma)
    sg = np.sin(gamma)

    # Construct the matrix exactly as defined in the mathematical formula
    R = np.array([
        [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
        [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
        [-sb, cb * sg, cb * cg]
    ])

    return R


def get_dh_rotation_matrix(alpha, theta):
    """
    Calculates the 3x3 rotation matrix for a single link
    using Craig's Modified DH convention.
    """
    c_th = cos(theta)
    s_th = sin(theta)
    c_alp = cos(alpha)
    s_alp = sin(alpha)

    # Extracting only the 3x3 rotation part (R) from the 4x4 transformation matrix (T)
    R = np.array([
        [c_th, -s_th, 0],
        [s_th * c_alp, c_th * c_alp, -s_alp],
        [s_th * s_alp, c_th * s_alp, c_alp]
    ])
    return R


def get_R_0_4_eval_zero(theta1, theta2, theta3):
    """
    Calculates the rotation matrix R_0_4 evaluated at theta_4 = 0.
    Uses the DH parameters from the manipulator's table.

    Parameters:
    theta1, theta2, theta3 : float : The solved joint angles in radians

    Returns:
    numpy.ndarray : A 3x3 rotation matrix
    """

    # Joint 1: alpha_0 = 0, theta_1
    R_0_1 = get_dh_rotation_matrix(0.0, theta1)

    # Joint 2: alpha_1 = pi/2, theta_2
    R_1_2 = get_dh_rotation_matrix(pi / 2, theta2)

    # Joint 3: alpha_2 = 0, theta_3
    R_2_3 = get_dh_rotation_matrix(0.0, theta3)

    # Joint 4 evaluated at theta_4 = 0: alpha_3 = pi/2, theta_4 = 0
    R_3_4_zero = get_dh_rotation_matrix(pi / 2, 0.0)

    # Multiply the matrices together: R_0_4 = R_0_1 * R_1_2 * R_2_3 * R_3_4
    R_0_2 = np.dot(R_0_1, R_1_2)
    R_0_3 = np.dot(R_0_2, R_2_3)
    R_0_4_zero = np.dot(R_0_3, R_3_4_zero)

    return R_0_4_zero

if __name__ == "__main__":

    x,y,z = [0.345, 0.0, 0.537]
    position = [x,y,z]
    alpha = np.arctan2(y,x)
    orientation = [0, 0, np.pi]

    joints = inverse_kinematics(position, orientation)

    print("--- Calculated Joint Angles (Theta) ---")
    print(f"theta1 = {joints[0]:.2f}, theta2 = {joints[1]:.2f}, theta3 = {joints[2]:.2f}, theta4 = {joints[3]:.2f}, theta5 = {joints[4]:.2f}, theta6 = {joints[5]:.2f}")
    # print(f"{joints[0]:.3f}, {joints[1]:.3f}, {joints[2]:.3f}, {joints[3]:.3f}, {joints[4]:.3f}, {joints[5]:.3f}")