# Autonomous Robot Arm for Weed Detection and Removal

**Bachelor Thesis Project** | Aarhus University (AU)  
A ROS2-based autonomous agricultural robot system for real-time weed detection, approach, and extraction using a custom 5-DOF robot manipulator.

## Project Overview

This repository contains the high-level ROS2 control software for an autonomous robot manipulator designed to perform precision agriculture tasks, specifically targeted at autonomous weed removal. The system combines:

- Vision-based weed detection using RGB-D cameras (OAK-D S2)
- Real-time inverse kinematics for Cartesian control
- Autonomous task execution via a finite state machine
- Adaptive grasping and extraction with feedback control

**Note:** This repository contains the high-level ROS 2 software. Low-level hardware actuation (motor control, gripper drivers, encoder feedback) is managed by a separate Raspberry Pi-based repository.

### Core Components

| Package | Purpose |
|---------|---------|
| **robot_arm_control** | Primary control logic, kinematics, and task execution |
| **robot_arm_description** | URDF model and launch configurations |
| **orange_detections** | Vision pipeline for detecting an orange test spike using HSV filtering |
| **spatial_detector**  | Spatial visualization and republishing of NN weed species detections from the OAK-D camera |
| **collision_avoidance** | Path planning using A* and trajectory smoothing (not yet implemented) |


## Main Components

### Task Controller (FSM)
**File:** [robot_arm_control/task_controllerV2.py](src/robot_arm_control/robot_arm_control/task_controllerV2.py)

The central orchestrator executing a finite state machine:

```
STARTUP → SCAN_SWEEP ↔ SCAN_CONFIRM → ALIGNING_XY → FINAL_SCAN 
  → APPROACHING → PLUNGING → GRASPING → EXTRACTING → DONE → PAUSE → STARTUP
```

### Kinematics
**Files:** 
- [robot_arm_control/InverseKinematics.py](src/robot_arm_control/robot_arm_control/InverseKinematics.py)
- [robot_arm_control/ForwardKinematics.py](src/robot_arm_control/robot_arm_control/ForwardKinematics.py)

**Forward Kinematics:**
- 5 revolute joints using Craig's modified Denavit-Hartenberg convention
- Returns transformation matrix

**Inverse Kinematics:**
- Geometric solution for 6-DOF pose (X, Y, Z, Roll, Pitch, Yaw)
- Gimbal lock detection and recovery
- Workspace validation and safety checks

### Vision System


**Files:**
- [robot_arm_control/VisionTransformV2.py](src/robot_arm_control/robot_arm_control/VisionTransformV2.py) — Camera-to-base frame transforms
- [src/spatial_detector/](src/spatial_detector/) — Spatial detection visualization and republishing
 
The vision system uses a custom made Neural Network (NN) to detect and calculate spatial coordinates of weeds. 

**Detection Pipeline:**
1. OAK-D S2 RGB-D camera captures color and depth
2. Camera's neural network classifies weed species (runs on camera hardware)
3. Spatial detections published via `/oak/nn/spatial_detections`
4. `spatial_detector` visualizes and republishes detection poses
5. Camera-to-base transform converts detections to robot base frame
6. IMU compensation: Uses camera's built-in IMU to correct for mechanical backlash

## System Workflow

### Single Weed Removal Cycle

1. **Startup** → Open gripper, move to scanning pose
2. **Scanning** → Execute arc motion with continuous weed detection (single frame mode for speed)
3. **Detection** → HSV filter identifies weed/orange target
4. **Confirmation** → Freeze arm, collect 30-frame high-quality measurement
5. **Aligning** → Move camera directly above detected weed (maintains Z-height)
6. **Refinement Scan** → 30-frame measurement at new optimal position for extracting precise coordinates
7. **Approach** → Vertical descent to soil surface height
8. **Plunging** → Lower gripper 4cm into soil
9. **Grasping** → Close gripper, wait 2 seconds for mechanical settling
10. **Extracting** → Pull straight up to safe height
11. **Release** → Return to scanning position, open gripper
12. **Logging** → Record task outcome to CSV, await user feedback

## Getting Started

### Prerequisites
- ROS2 (Jazzy)
- Python 3.10+
- OpenCV with Python bindings
- OAK-D S2 camera with ROS 2 wrapper
- Raspberry Pi repository running low-level hardware control

### Installation

1. **Clone the repository:**
   ```bash
   cd ~/robot_arm_ws
   ```

2. **Install dependencies:**
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. **Build the workspace:**
   ```bash
   colcon build
   source install/setup.bash
   ```

### Running the System

**Start visualization and vision:**
```bash
ros2 launch robot_arm_description rviz_robot_cam.launch.py
```

This launches:
- RVIZ for visualization
- Robot state publisher (transforms)
- Joint state publisher
- Launches Camera 
- Detection Republisher

**In separate terminals, manually launch the control nodes:**
```bash
# Terminal 1: Task controller (main FSM)
ros2 run robot_arm_control task_controllerV2

# Terminal 2: Inverse kinematics streamer
ros2 run robot_arm_control live_ik_streamer
```
