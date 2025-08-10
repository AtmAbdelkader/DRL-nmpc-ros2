#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file_name = LaunchConfiguration('world_name', default='td3.world')

    world = PathJoinSubstitution([
        get_package_share_directory('td3'),
        'worlds',
        world_file_name
    ])
    launch_file_dir = os.path.join(get_package_share_directory('td3'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    rviz_file = os.path.join(get_package_share_directory('td3'), 'launch', 'pioneer3dx.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'world_name',
            default_value='TD3.world',
            description='Name of the world file to load'
        ),

        # Start Gazebo server (headless)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),

        # ❌ تم تعطيل gzclient GUI
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        #     ),
        # ),

        # Training node
        Node(package='td3',
             executable='train_robot.py',
             output='screen'
        ),

        # Robot state publisher
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # RViz2 visualization
        Node(package='rviz2',
            executable='rviz2',
            name='rviz2',  
            arguments=['-d', rviz_file],
            output='screen'
        ),
    ])
