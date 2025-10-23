import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R


class StaticExtrinPublisher(Node):
    def __init__(self):
        super().__init__('static_extrin_publisher')

        self.declare_parameter('frames.camera', 'camera_link')
        self.declare_parameter('frames.lidar',  'velodyne')
        self.declare_parameter('extrin_calib.Rcl', [1.0, 0.0, 0.0,
                                                    0.0, 1.0, 0.0,
                                                    0.0, 0.0, 1.0])
        self.declare_parameter('extrin_calib.Pcl', [0.0, 0.0, 0.0])

        cam = self.get_parameter('frames.camera').get_parameter_value().string_value
        lid = self.get_parameter('frames.lidar').get_parameter_value().string_value
        Rcl_raw = self.get_parameter('extrin_calib.Rcl').get_parameter_value().double_array_value
        Pcl_raw = self.get_parameter('extrin_calib.Pcl').get_parameter_value().double_array_value

        Rcl = np.array(Rcl_raw, dtype=float).reshape(3, 3)
        Pcl = np.array(Pcl_raw, dtype=float).reshape(3)

        qx, qy, qz, qw = R.from_matrix(Rcl).as_quat()

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = cam
        t.child_frame_id  = lid
        t.transform.translation.x = float(Pcl[0])
        t.transform.translation.y = float(Pcl[1])
        t.transform.translation.z = float(Pcl[2])
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)

        self.br = StaticTransformBroadcaster(self)
        self.br.sendTransform(t)
        self.get_logger().info(f"Published static TF {cam} -> {lid}")

def main():
    rclpy.init()
    n = StaticExtrinPublisher()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
