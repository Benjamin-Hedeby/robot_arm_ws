#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # 1. Get your specific package directory
    pkg_share = FindPackageShare('robot_arm_description')

    # ================== Declare Launch Arguments =================== #
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='false', # Keep false unless running alongside Gazebo
        description='Use simulation (Gazebo) clock if true')

    declare_jsp_gui_cmd = DeclareLaunchArgument(
        'jsp_gui', 
        default_value='True',
        description='Flag to enable joint_state_publisher_gui')
        
    declare_urdf_model_cmd = DeclareLaunchArgument(
        'urdf_model',
        default_value=PathJoinSubstitution([pkg_share, 'robot.urdf']), # Pointing directly to the plain URDF
        description='Absolute path to robot URDF file')
    
    # ================== Robot Description Setup =================== #
    # Process the URDF file (the xacro command safely reads standard URDFs too)
    robot_description_content = ParameterValue(
        Command(['xacro ', LaunchConfiguration('urdf_model')]),
        value_type=str
    )

    # ================== Node Definitions =================== #
    # ---- Robot State Publisher ----#
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': robot_description_content,
        }]
    )

    # --------- Joint State Publisher GUI Node ----------
    start_joint_state_publisher_gui_cmd = Node(
        condition=IfCondition(PythonExpression([LaunchConfiguration('jsp_gui')])),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # ---- RViz2 Node ----
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        # Omitted the '-d' config argument so it launches cleanly without a saved config file
    )

    # ================== Create Launch Description =================== #
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_jsp_gui_cmd)
    ld.add_action(declare_urdf_model_cmd)

    # Add nodes to the launch description
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_joint_state_publisher_gui_cmd)
    ld.add_action(start_rviz_cmd)

    return ld