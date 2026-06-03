from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # ================== 1. Define Paths ==================
    pkg_share = FindPackageShare('robot_arm_description')
    
    urdf_path = PathJoinSubstitution([pkg_share, 'robot.urdf'])
    
    # rviz_config_path = PathJoinSubstitution([pkg_share, 'rviz', 'display.rviz'])
    
    # Point to your custom camera YAML config
    camera_config_path = PathJoinSubstitution([pkg_share, 'params', 'camera_yolo8s.yaml'])

    # Process the URDF
    robot_description_content = ParameterValue(
        Command(['xacro ', urdf_path]),
        value_type=str
    )

    # ================== 2. Robot Nodes ==================
    
    # Robot State Publisher (Calculates 3D coordinates)
    start_rsp_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}],

        remappings=[('/robot_description', '/arm_description')]
    )

    # Headless Joint State Publisher (Publishes 0.0 so RViz doesn't error on boot)
    start_jsp_cmd = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'robot_description': robot_description_content}],

        remappings=[('/robot_description', '/arm_description')] # <--- Add it here too!
    )

    # RViz2 Node
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        # arguments=['-d', rviz_config_path] # Uncomment this line if you saved a display.rviz file!
    )

    # ================== 3. Camera Driver ==================
    
    start_camera_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('depthai_ros_driver'), 
                'launch', 
                'camera.launch.py'
            ])
        ),
        launch_arguments={
            'name': 'oak',
            'parent_frame': 'oak_d_s2',   

            # --- TRANSLATION (in meters) ---
            'cam_pos_x': '0.0',
            'cam_pos_y': '0.0',
            'cam_pos_z': '-0.02307',
            
            # --- ROTATION (in radians) ---
            'cam_roll': '1.5708',  # Example: 90 degrees roll
            'cam_pitch': '1.5708', 
            'cam_yaw': '0.0',

            'publish_urdf': 'true',      
            'use_urdf': 'false', 
            'publish_tf': 'false',
            'params_file': camera_config_path
        }.items()
    )

    start_overlay_cmd = Node(
        package='spatial_detector',
        executable='spatial_overlay',
        name='spatial_overlay',
        output='screen'
    )

    start_visualizer_cmd = Node(
        package='spatial_detector',
        executable='spatial_visualizer',
        name='spatial_visualizer',
        output='screen'
    )

    start_detection_republisher_cmd = Node(
        package='controller_joint_test',
        executable='detections_republish',
        name='detections_republish',
        output='screen'
    )

    # ================== 4. Return Everything ==================
    return LaunchDescription([
        start_rsp_cmd,
        # start_jsp_cmd,
        start_rviz_cmd,
        start_camera_cmd,
        start_overlay_cmd,
        start_visualizer_cmd,
        start_detection_republisher_cmd
    ])