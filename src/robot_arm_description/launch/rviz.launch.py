from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    # 1. Point directly to your URDF file
    urdf_path = PathJoinSubstitution([
        FindPackageShare('robot_arm_description'), 
        'robot.urdf'
    ])

    # 2. Process the URDF
    robot_description_content = ParameterValue(
        Command(['xacro ', urdf_path]),
        value_type=str
    )

    # 3. Robot State Publisher (This automatically listens to the /joint_states topic)
    start_rsp_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    # 4. RViz Node
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        output='screen'
    )

    # Return the clean launch description
    return LaunchDescription([
        start_rsp_cmd,
        start_rviz_cmd,
    ])