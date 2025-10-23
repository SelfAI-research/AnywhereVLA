import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
import time
import numpy as np
from robot_model import RobotModel

# How often we publish odometry (Hz)
ODOMETRY_RATE = 2.0   # ~13 ms per update
# How often we send velocity commands to the motors (Hz)
COMMAND_RATE  = 40.0   # ~25 ms per update
# How often we restart motors to recover from blockage
RECOVERY_RATE = 6    # ~26 ms for one motor, stops that motor for a moment

odometry_times = [0]
command_times = [0]
recover_times = [0]

class MainMotorsNode(Node):
    """
    ROS2 node that bridges cmd_vel → MotorHardwareInterface → odometry.

    Subscribes to /cmd_vel (geometry_msgs/Twist), forwards commands
    at COMMAND_RATE Hz to the motors, and publishes /odom
    (nav_msgs/Odometry) at ODOMETRY_RATE Hz based on dead-reckoned pose.
    """

    def __init__(self):
        super().__init__('main_motors_node')

        self.get_logger().info('Initializing MotorHardwareInterface…')
        self.robot = RobotModel()
        self.get_logger().info('RobotModel ready, starting ROS interfaces')

        # Last received command [linear, angular]
        self.cmd_speed = [0.0, 0.0]

        # Subscribers & publishers
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel',
            self.cmd_vel_callback,
            qos_profile=2
        )
        self.joy_sub = self.create_subscription(
            Twist, '/cmd_vel_joy',
            self.cmd_joy_vel_callback,
            qos_profile=2
        )
        self.odom_pub = self.create_publisher(
            Odometry, '/odom',
            qos_profile=1
        )

        # recovery
        self.recovery_motor_index = 0
        self.last_joy_command = 0
        self.joystick_priority_time = 3.0  # seconds

        # Timers
        self.create_timer(1.0 / COMMAND_RATE, self.send_command_to_motors)
        self.create_timer(1.0 / ODOMETRY_RATE, self.read_odometry)
        self.create_timer(1.0 / RECOVERY_RATE, self.motors_recovery)

    def cmd_vel_callback(self, msg: Twist):
        """
        Callback for incoming velocity commands.
        Stores the latest linear.x and angular.z components.
        """
        cur_time = time.time()
        if (cur_time - self.last_joy_command) < self.joystick_priority_time:
            self.get_logger().info(f"Do not set cmd_vel speed as gamepad is in priority")
            return
        lin = msg.linear.x
        ang = msg.angular.z
        self.cmd_speed = [lin, ang]
        self.get_logger().info(f"Received cmd_vel: linear={lin:.3f}  angular={ang:.3f}")

    def cmd_joy_vel_callback(self, msg: Twist):
        """
        Callback for incoming gamepad velocity commands.
        Stores the latest linear.x and angular.z components.
        """
        self.last_joy_command = time.time()
        lin = msg.linear.x
        ang = msg.angular.z
        self.cmd_speed = [lin, ang]
        self.get_logger().info(f"Received cmd_vel_joy: linear={lin:.3f}  angular={ang:.3f}")

    def motors_recovery(self):
        t1 = time.perf_counter()
        self.get_logger().debug(f"Restart {self.recovery_motor_index} motor")
        self.robot.motors_recovery(self.recovery_motor_index)
        self.recovery_motor_index = (self.recovery_motor_index + 1) % 2
        t2 = time.perf_counter()
        recover_times.append(t2-t1)


    def send_command_to_motors(self):
        """
        Periodic timer callback that actually sends the stored cmd_vel
        to the MotorHardwareInterface.
        """
        t1 = time.perf_counter()
        lin, ang = self.cmd_speed
        self.robot.set_velocity(lin, ang)
        t2 = time.perf_counter()
        command_times.append(t2-t1)

    def read_odometry(self):
        """
        Periodic timer callback that updates the RobotModel, builds
        a nav_msgs/Odometry message, and publishes it.
        """
        t1 = time.perf_counter()
        # 1) Update internal pose & velocity
        self.robot.update()
        state = self.robot.get_state()

        # 2) Fill Odometry message
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        # Pose
        odom.pose.pose.position.x = state.x
        odom.pose.pose.position.y = state.y
        quat = R.from_euler('z', state.theta).as_quat()  # returns [x, y, z, w]
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        # Twist
        odom.twist.twist.linear.x  = state.linear_vel
        odom.twist.twist.angular.z = state.angular_vel

        # 3) Publish
        self.odom_pub.publish(odom)
        t2 = time.perf_counter()
        odometry_times.append(t2-t1)
        # print(f"Odom times:    {np.mean(odometry_times):.6f} +- {np.std(odometry_times):.6f}")
        # print(f"Command times: {np.mean(command_times):.6f} +- {np.std(command_times):.6f}")
        # print(f"Recover times: {np.mean(recover_times):.6f} +- {np.std(recover_times):.6f}")
        # time_t = np.mean(recover_times[-10:]) * RECOVERY_RATE + np.mean(odometry_times[-10:]) * ODOMETRY_RATE + np.mean(command_times[-30:]) * COMMAND_RATE
        # print(f"Full time: {time_t:.6f} s")


def main():
    """
    Entry point — initializes ROS2, spins the node, and ensures
    motors are stopped on shutdown.
    """
    rclpy.init()
    node = MainMotorsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt — stopping robot")
        node.robot.set_velocity(0.0, 0.0)
    finally:
        node.get_logger().info('Shutting down motors node')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
