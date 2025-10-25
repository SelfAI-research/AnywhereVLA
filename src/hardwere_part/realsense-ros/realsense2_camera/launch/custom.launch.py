from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera_d435',
            namespace='d435',
            parameters=[{
                'camera_name': 'd435',
                'device_type': 'd435'
            }]
        ),
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera_t265',
            namespace='t265',
            parameters=[{
                'camera_name': 't265',
                'device_type': 't265',
                'enable_pose': True
            }]
        )
    ])