import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    os.environ['TURTLEBOT3_MODEL'] = 'waffle'

    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    world_path = os.path.expanduser('~/Desktop/2026-Jay/multi-client-communication-system/v6.0.0/gazebo/lab_world')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_path}.items(),
    )

    # Robot 1 (Guide Waffle): Fixed pose (Facing East, yaw = 0.0)
    spawn_robot1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot1',
            '-file', os.path.join(turtlebot3_gazebo_dir, 'models', 'turtlebot3_waffle', 'model.sdf'),
            '-x', '0.092', '-y', '5.528', '-z', '0.01', '-Y', '0.0',
            '-robot_namespace', '/robot1'
        ],
        output='screen'
    )

    # Robot 2 (Burger): Corridor Right (Facing West)
    spawn_robot2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot2',
            '-file', os.path.join(turtlebot3_gazebo_dir, 'models', 'turtlebot3_burger', 'model.sdf'),
            '-x', '9.695', '-y', '1.035', '-z', '0.01', '-Y', '3.140',
            '-robot_namespace', '/robot2'
        ],
        output='screen'
    )

    # Robot 3 (Burger): Lower Corridor (Facing North)
    spawn_robot3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot3',
            '-file', os.path.join(turtlebot3_gazebo_dir, 'models', 'turtlebot3_burger', 'model.sdf'),
            '-x', '8.128', '-y', '-5.582', '-z', '0.01', '-Y', '1.570',
            '-robot_namespace', '/robot3'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_robot1,
        spawn_robot2,
        spawn_robot3
    ])
