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
    # /arm/robot_description keeps this off kridtbot's own /robot_description
    # when both run on the same ROS domain (Mini PC integration).
    start_rsp_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}],

        remappings=[('/robot_description', '/arm/robot_description')]
    )

    # Headless Joint State Publisher (Publishes 0.0 so RViz doesn't error on boot)
    start_jsp_cmd = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'robot_description': robot_description_content}],

        remappings=[('/robot_description', '/arm/robot_description')] # <--- Add it here too!
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
            # 'oak_arm' (not the depthai default 'oak') so this doesn't collide
            # with kridtbot's own OAK-D, which also uses the default 'oak' name
            # (see kridtbot's camera_params.yaml i_tf_camera_name).
            'name': 'oak_arm',
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

    # Remap table shared by the vision-adjacent nodes below: the camera now
    # publishes under /oak_arm/... (see start_camera_cmd), and every other
    # absolute topic these nodes use is hardcoded with a leading slash in
    # their source, so namespacing the launch won't touch it -- explicit
    # remaps are the only thing that works.
    oak_remaps = [
        ('/oak/rgb/camera_info', '/oak_arm/rgb/camera_info'),
        ('/oak/rgb/image_rect', '/oak_arm/rgb/image_rect'),
        ('/oak/nn/spatial_detections', '/oak_arm/nn/spatial_detections'),
    ]

    start_overlay_cmd = Node(
        package='spatial_detector',
        executable='spatial_overlay',
        name='spatial_overlay',
        output='screen',
        remappings=oak_remaps,
    )

    start_visualizer_cmd = Node(
        package='spatial_detector',
        executable='spatial_visualizer',
        name='spatial_visualizer',
        output='screen',
        remappings=oak_remaps,
    )

    # NOTE: package was 'controller_joint_test', which has never actually
    # registered a 'detections_republish' executable (only 'controller_joint_test'
    # itself) -- this Node has been broken since it was written. The real
    # detections_republish lives in robot_arm_control; fixed here.
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