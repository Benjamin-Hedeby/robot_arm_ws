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

**On kridtbot's Mini PC (integrated with the rover):**
```bash
ros2 launch robot_arm_description arm_bringup.launch.py
```
Headless bringup for real operation on the rover. Every topic here is
remapped under `/arm/...` (camera under `/oak_arm/...`) so it can't collide
with kridtbot's own rover topics when both run on the same ROS domain — see
the comments at the top of `arm_bringup.launch.py`. This assumes the
Raspberry Pi is reachable on the same `ROS_DOMAIN_ID` and already running
`arm_control`'s `arm.launch.py` (also namespaced `/arm`), which is the
canonical source of `/arm/joint_states` and `/arm/robot_description`.

**On a dev machine, for bench testing/visualization (not namespaced):**
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
ros2 run robot_arm_control task_controller

# Terminal 2: Inverse kinematics streamer
ros2 run robot_arm_control live_ik_streamer
```

Run this way (no remaps), these talk on the old unnamespaced topics
(`/joint_states`, `/arm_controller/commands`, ...) — fine for isolated bench
testing against `rviz_robot_cam.launch.py`, but they will **not** reach the
Pi once it's running the namespaced `arm.launch.py`. Use
`arm_bringup.launch.py` for anything talking to real hardware.

**Same two terminals, but against real hardware:** add the `/arm` (and
`/oak_arm`) remaps by hand so each node reaches the namespaced topics the Pi
and camera actually use — this is the manual equivalent of what
`arm_bringup.launch.py` does for you automatically:
```bash
# Terminal 1: Task controller (main FSM)
ros2 run robot_arm_control task_controller --ros-args \
  -r /desired_tcp_pose_euler:=/arm/desired_tcp_pose_euler \
  -r /gripper_open_close_cmd:=/arm/gripper_open_close_cmd \
  -r /trigger_measurement:=/arm/trigger_measurement \
  -r /joint_states:=/arm/joint_states \
  -r /weed_location_cam_frame:=/arm/weed_location_cam_frame \
  -r /oak/imu/data:=/oak_arm/imu/data

# Terminal 2: Inverse kinematics streamer
ros2 run robot_arm_control live_ik_streamer --ros-args \
  -r /desired_tcp_pose_euler:=/arm/desired_tcp_pose_euler \
  -r /arm_controller/commands:=/arm/arm_controller/commands
```
