import numpy as np

# ==========================================
# 1. PHYSICAL ROBOT DH PARAMETERS (Meters)
# ==========================================
A1 = 0.0435
A2 = 0.300
D4 = 0.389
D6 = 0.203

# Craig's Modified DH Table: [alpha_(i-1), a_(i-1), d_i]
DH_TABLE = [
    [0.0,         0.0,  0.0],  # Frame 1
    [np.pi / 2,   A1,   0.0],  # Frame 2
    [0.0,         A2,   0.0],  # Frame 3
    [np.pi / 2,   0.0,  D4],   # Frame 4
    [-np.pi / 2,  0.0,  0.0],  # Frame 5
    [np.pi / 2,   0.0,  D6]    # Frame 6 (End-Effector)
]

# ==========================================
# 2. SAFETY & JOINT LIMITS (Radians)
# ==========================================
JOINT_LIMITS = [
    [-3.12, 3.12],   # Joint 1 (Base)
    [-0.87, 1.95],   # Joint 2 (Shoulder)
    [-1.83, 1.83],   # Joint 3 (Elbow)
    [-2.09, 2.09],   # Joint 4 (Wrist 1)
    [-2.0, 2.0]      # Joint 5 (Wrist 2)
]

# ==========================================
# 3. CAMERA VISION OFFSETS (Meters)
# ==========================================
# The physical offset of the camera origin relative to the origin of frame 6
CAMERA_OFFSET_X = 0.00
CAMERA_OFFSET_Y = 0.051
CAMERA_OFFSET_Z = -0.136