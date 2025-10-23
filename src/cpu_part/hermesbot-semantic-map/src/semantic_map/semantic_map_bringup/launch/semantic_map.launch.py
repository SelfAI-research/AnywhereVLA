import yaml
from launch import LaunchDescription, LaunchContext, logging
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg_arg = DeclareLaunchArgument(
        "config", default_value=PathJoinSubstitution(
            [FindPackageShare("semantic_map"), "config"]
        ),
        description="Path to semantic_map config",
    )
    cfg_path = LaunchConfiguration("config")

    def setup(context: LaunchContext, *_):
        path = context.perform_substitution(cfg_path)
        log = logging.get_logger("semantic_map_launch")
        log.info(f"Loaded config from: {path}")

        nodes = [
            Node(
                package="semantic_map", executable="semantic_map_3d",
                name="detect_3d_node_lidar", output="screen",
                parameters=[{"config_folder": path}],
            ),
        ]


        return nodes

    return LaunchDescription([cfg_arg, OpaqueFunction(function=setup)])
