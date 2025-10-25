import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from evdev import InputDevice, ecodes
from glob import glob

LINEAR_SPEED = 0.8
ANGULAR_SPEED = 1.8
DEADZONE = 0.06  # stick values within ±DEADZONE → treated as zero

class GamepadNode(Node):
    def __init__(self):
        super().__init__('gamepad_node')
        self.pub = self.create_publisher(Twist, 'cmd_vel_joy', 1)
        self.enabled = True
        self.speed = [0.0, 0.0]

        self.get_logger().info('Waiting for Logitech F710 gamepad...')
        while rclpy.ok():
            try:
                path = glob('/dev/input/by-id/usb-Logitech_Wireless_Gamepad_F710_*-event-joystick')[0]
                self.gamepad = InputDevice(path)
                self.get_logger().info(f'Connected to {path}')
                break
            except Exception:
                self.get_logger().warn('Gamepad not found, retrying in 1s...')
                time.sleep(1.0)

    def publish_speed(self):
        self.get_logger().debug(f'Send speed: linear={self.speed[0]:.3f}, angular={self.speed[1]:.3f}')
        msg = Twist()
        msg.linear.x = self.speed[0]
        msg.angular.z = self.speed[1]
        self.pub.publish(msg)

    def spin(self):
        for event in self.gamepad.read_loop():
            if not rclpy.ok():
                break

            # Toggle ON/OFF with X button (BTN_NORTH)
            if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_NORTH and event.value == 1:
                self.enabled = not self.enabled
                state = 'ENABLED' if self.enabled else 'DISABLED'
                self.get_logger().info(f'Gamepad control {state}')
                if not self.enabled:
                    # stop robot when turning off
                    self.speed = [0.0, 0.0]
                    self.publish_speed()
                continue

            # Only process sticks when enabled
            if not self.enabled or event.type != ecodes.EV_ABS:
                continue

            code = ecodes.bytype[event.type][event.code]
            norm = event.value / 32767.0
            # deadzone
            if abs(norm) < DEADZONE:
                norm = 0.0

            if code == 'ABS_Y':
                # left stick vertical axis: forward/back
                self.speed[0] = -norm * LINEAR_SPEED
                self.publish_speed()

            elif code == 'ABS_RX':
                # right stick horizontal axis: turn
                self.speed[1] = -norm * ANGULAR_SPEED
                self.publish_speed()

def main():
    rclpy.init()
    node = GamepadNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down')
        node.destroy_node()

if __name__ == '__main__':
    main()
