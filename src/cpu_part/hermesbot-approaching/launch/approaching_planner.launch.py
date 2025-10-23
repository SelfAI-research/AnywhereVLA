from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description() -> LaunchDescription:
    params = DeclareLaunchArgument("params_file", default_value="config/params.yaml")
    run = ExecuteProcess(
        cmd=[
            "python3", "src/approach_planner_node.py",
            "--ros-args", "--params-file", config/params.yaml
        ],
        env={"PYTHONUNBUFFERED": "1"},
        output="screen",
    )
    return LaunchDescription([params, run])
