# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Launch the velodyne driver, pointcloud, and laserscan nodes in a composable container with custom configuration for the driver node."""

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # Path to the velodyne_driver configuration file
    driver_share_dir = get_package_share_directory('velodyne_driver')
    driver_params_file = os.path.join(driver_share_dir, 'config', 'VLP16-velodyne_driver_node-params.yaml')

    # Load default parameters for velodyne_driver_node
    with open(driver_params_file, 'r') as f:
        driver_params = yaml.safe_load(f)['velodyne_driver_node']['ros__parameters']

    # Override default parameters with custom values
    driver_params.update({
        "device_ip": "192.168.3.201",
        "host": "0.0.0.0",
        "port": 2368,
        "model": "VLP16",
        "rpm": 600.0,
        "frame_id": "velodyne"
    })

    # Path to the velodyne_pointcloud configuration file
    convert_share_dir = get_package_share_directory('velodyne_pointcloud')
    convert_params_file = os.path.join(convert_share_dir, 'config', 'VLP16-velodyne_transform_node-params.yaml')

    # Load parameters for velodyne_transform_node
    with open(convert_params_file, 'r') as f:
        convert_params = yaml.safe_load(f)['velodyne_transform_node']['ros__parameters']
    convert_params['calibration'] = os.path.join(convert_share_dir, 'params', 'VLP16db.yaml')

    # Path to the velodyne_laserscan configuration file
    laserscan_share_dir = get_package_share_directory('velodyne_laserscan')
    laserscan_params_file = os.path.join(laserscan_share_dir, 'config', 'default-velodyne_laserscan_node-params.yaml')

    # Load parameters for velodyne_laserscan_node
    with open(laserscan_params_file, 'r') as f:
        laserscan_params = yaml.safe_load(f)['velodyne_laserscan_node']['ros__parameters']

    # Define the ComposableNodeContainer
    container = ComposableNodeContainer(
        name='velodyne_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='velodyne_driver',
                plugin='velodyne_driver::VelodyneDriver',
                name='velodyne_driver_node',
                parameters=[driver_params]
            ),
            ComposableNode(
                package='velodyne_pointcloud',
                plugin='velodyne_pointcloud::Transform',
                name='velodyne_transform_node',
                parameters=[convert_params]
            ),
            ComposableNode(
                package='velodyne_laserscan',
                plugin='velodyne_laserscan::VelodyneLaserScan',
                name='velodyne_laserscan_node',
                parameters=[laserscan_params]
            ),
        ],
        output='both',
    )

    return LaunchDescription([container])