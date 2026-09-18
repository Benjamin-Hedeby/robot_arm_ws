"""Headless bringup for the arm's high-level software (task controller, IK,
vision) on kridtbot's Mini PC.

Unlike rviz_robot_cam.launch.py (dev/visualization, meant for a laptop with
a monitor attached), this has no RViz and no robot_state_publisher of its
own -- the canonical one for the arm runs on the Raspberry Pi (namespaced
'arm' in RPI-Code's arm.launch.py) and is reachable over the network once
both hosts share a ROS_DOMAIN_ID.

Every topic here is remapped under /arm/... (or /oak_arm/... for the
camera) so it can't collide with kridtbot's own rover topics when both run
on the Mini PC at once. The source files themselves (task_controllerV2.py,
live_ik_streamer.py, detections_republish.py) hardcode absolute topic names
with a leading slash, so a namespace push on these Node actions would NOT
reach them -- explicit remappings are the only thing that works.

fake_hardware:=true replaces the camera stack with robot_arm_control's
fake_arm (which also stands in for the Pi), so the task controller can be
tested on any machine without the robot. Don't use it with the Pi running
on the same ROS domain: both would publish /arm/joint_states.

Nothing moves until a cycle is requested:
  ros2 action send_goal /arm/remove_weed robot_arm_interfaces/action/RemoveWeed "{}" --feedback
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    fake_hardware = LaunchConfiguration('fake_hardware')
    pkg_share = FindPackageShare('robot_arm_description')
    camera_config_path = PathJoinSubstitution([pkg_share, 'params', 'camera_yolo8s.yaml'])

    # ================== Camera Driver ==================
    # Same pose/config as rviz_robot_cam.launch.py's camera include, minus
    # the dev-only bits (RViz, joint_state_publisher_gui).
    camera_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('depthai_ros_driver'),
                'launch',
                'camera.launch.py'
            ])
        ),
        launch_arguments={
            # 'oak_arm', not depthai's default 'oak' -- kridtbot's own OAK-D
            # also uses the default 'oak' name (see camera_params.yaml
            # i_tf_camera_name in the kridtbot repo).
            'name': 'oak_arm',
            'parent_frame': 'oak_d_s2',

            'cam_pos_x': '0.0',
            'cam_pos_y': '0.0',
            'cam_pos_z': '-0.02307',

            'cam_roll': '1.5708',
            'cam_pitch': '1.5708',
            'cam_yaw': '0.0',

            'publish_urdf': 'true',
            'use_urdf': 'false',
            'publish_tf': 'false',
            'params_file': camera_config_path
        }.items()
    )

    # depthai's camera URDF publisher has no namespace or remap, so it would
    # publish on /robot_description -- the topic kridtbot's controller manager
    # (and Gazebo's spawner) read the rover's URDF from. The group scopes this
    # remap to the camera's nodes only; it reaches the publisher even though
    # depthai loads it as a component inside the camera container (verified).
    start_camera_cmd = GroupAction([
        SetRemap(src='/robot_description', dst='/oak_arm/robot_description'),
        camera_include,
    ], condition=UnlessCondition(fake_hardware))

    oak_remaps = [
        ('/oak/rgb/camera_info', '/oak_arm/rgb/camera_info'),
        ('/oak/rgb/image_rect', '/oak_arm/rgb/image_rect'),
        ('/oak/nn/spatial_detections', '/oak_arm/nn/spatial_detections'),
        ('/oak/imu/data', '/oak_arm/imu/data'),
    ]

    # ================== Vision ==================
    start_detection_republisher_cmd = Node(
        package='robot_arm_control',
        executable='detections_republish',
        name='detections_republish',
        output='screen',
        remappings=oak_remaps + [
            ('/range', '/arm/range'),
            ('/trigger_measurement', '/arm/trigger_measurement'),
            ('/weed_location_cam_frame', '/arm/weed_location_cam_frame'),
        ],
        condition=UnlessCondition(fake_hardware),
    )

    # Optional, for monitoring/debugging via rqt or rviz -- not required by
    # the task controller's control loop itself.
    start_overlay_cmd = Node(
        package='spatial_detector',
        executable='spatial_overlay',
        name='spatial_overlay',
        output='screen',
        remappings=oak_remaps,
        condition=UnlessCondition(fake_hardware),
    )

    start_visualizer_cmd = Node(
        package='spatial_detector',
        executable='spatial_visualizer',
        name='spatial_visualizer',
        output='screen',
        remappings=oak_remaps,
        condition=UnlessCondition(fake_hardware),
    )

    # ================== Fake hardware (testing only) ==================
    start_fake_arm_cmd = Node(
        package='robot_arm_control',
        executable='fake_arm',
        name='fake_arm',
        namespace='arm',
        output='screen',
        condition=IfCondition(fake_hardware),
    )

    # ================== Task Controller (FSM) + IK ==================
    # The namespace only affects the RemoveWeed action (a relative name -> /arm/remove_weed);
    # its topics are absolute and remapped below.
    start_task_controller_cmd = Node(
        package='robot_arm_control',
        executable='task_controller',  # registered console_script -> task_controllerV2:main
        name='task_controller',
        namespace='arm',
        output='screen',
        remappings=[
            ('/desired_tcp_pose_euler', '/arm/desired_tcp_pose_euler'),
            ('/gripper_open_close_cmd', '/arm/gripper_open_close_cmd'),
            ('/trigger_measurement', '/arm/trigger_measurement'),
            ('/joint_states', '/arm/joint_states'),
            ('/weed_location_cam_frame', '/arm/weed_location_cam_frame'),
            ('/oak/imu/data', '/oak_arm/imu/data'),
        ],
    )

    start_live_ik_streamer_cmd = Node(
        package='robot_arm_control',
        executable='live_ik_streamer',
        name='live_ik_streamer',
        output='screen',
        remappings=[
            ('/desired_tcp_pose_euler', '/arm/desired_tcp_pose_euler'),
            ('/arm_controller/commands', '/arm/arm_controller/commands'),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'fake_hardware', default_value='false',
            description='Use fake_arm instead of the camera (and without the Pi) for testing'),
        start_camera_cmd,
        start_detection_republisher_cmd,
        start_overlay_cmd,
        start_visualizer_cmd,
        start_fake_arm_cmd,
        start_task_controller_cmd,
        start_live_ik_streamer_cmd,
    ])
