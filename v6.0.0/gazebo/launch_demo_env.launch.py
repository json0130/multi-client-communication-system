import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    os.environ['TURTLEBOT3_MODEL'] = 'waffle'

    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    here = os.path.dirname(os.path.abspath(__file__))
    world_path = os.path.join(here, 'lab_world')

    # Custom per-robot models (TurtleBot3 + a floating name-label link fixed-
    # jointed to base_link, so the label moves with the robot via physics —
    # see gazebo/models/*_label/model.sdf) live under gazebo/models/. Gazebo
    # resolves each model's own model://<dir>/materials/... URIs (used for
    # the label's texture) against GAZEBO_MODEL_PATH, so this directory has
    # to be on it — prepended, not replaced, so the stock model database
    # Gazebo already knows about (ground_plane, sun, etc.) still resolves.
    models_dir = os.path.join(here, 'models')
    os.environ['GAZEBO_MODEL_PATH'] = models_dir + os.pathsep + os.environ.get('GAZEBO_MODEL_PATH', '')

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
            '-file', os.path.join(models_dir, 'pepper_label', 'model.sdf'),
            '-x', '0.092', '-y', '5.528', '-z', '0.01', '-Y', '0.0',
            '-robot_namespace', '/robot1'
        ],
        output='screen'
    )

    # Robot 2 (Burger): Corridor Right (Facing West). Uses the custom
    # cardboard-cat shell (chatbox_test/) validated side-by-side against the
    # stock chatbox_label/ turtlebot earlier — now promoted to production.
    spawn_robot2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot2',
            '-file', os.path.join(models_dir, 'chatbox_test', 'model.sdf'),
            '-x', '9.695', '-y', '1.035', '-z', '0.01', '-Y', '3.140',
            '-robot_namespace', '/robot2'
        ],
        output='screen'
    )

    # Robot 3 (Burger): navel_01. Originally at (-3.0, -5.0) — the west bay,
    # only reachable from the south (the bay's north/west/east sides are
    # walled, see demo_script.py's GAZEBO_ROUTES comments on the
    # SILBOT/NAVEL edge). Moved to (0.25, -5.0) — the MIDDLE bay — live, in
    # the Gazebo GUI, after the guide repeatedly failed to reach the west
    # bay via a from-the-north approach. NOT yet confirmed which direction
    # the middle bay IS reachable from; it looks similarly enclosed from
    # the north (walled by Wall_38/Wall_40/Wall_41) in the world's wall
    # geometry. GAZEBO_ROUTES' (SILBOT, NAVEL)/(NAVEL, SILBOT)/(NAVEL, START)
    # edges need to be re-derived and live-tested against this new position
    # — see the "UNVERIFIED" notes there.
    # Uses the custom humanoid/beanie shell (navel_test/) validated
    # side-by-side against the stock navel_label/ turtlebot earlier — now
    # promoted to production.
    spawn_robot3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot3',
            '-file', os.path.join(models_dir, 'navel_test', 'model.sdf'),
            '-x', '0.25', '-y', '-5.0', '-z', '0.01', '-Y', '0.0',
            '-robot_namespace', '/robot3'
        ],
        output='screen'
    )

    # Robot 4 (Burger): silbot_01 — moved to what was navel_01's station.
    # Uses the custom tablet-face/hourglass shell (silbot_test/) validated
    # side-by-side against the stock silbot_label/ turtlebot earlier — now
    # promoted to production.
    spawn_robot4 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'robot4',
            '-file', os.path.join(models_dir, 'silbot_test', 'model.sdf'),
            '-x', '8.128', '-y', '-5.582', '-z', '0.01', '-Y', '1.570',
            '-robot_namespace', '/robot4'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_robot1,
        spawn_robot2,
        spawn_robot3,
        spawn_robot4,
    ])
