from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='velodyne_driver',
            executable='velodyne_driver_node',
            name='velodyne_driver_node',
            output='screen',
            parameters=[{
                "device_ip": "192.168.3.201",
                "host": "192.168.3.100",
                "port": 2368,
                "model": "VLP16",
                "rpm": 600.0,
                "frame_id": "velodyne"
            }]
        )
    ])