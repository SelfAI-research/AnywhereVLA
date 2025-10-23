import os
import glob
from pymodbus.client.sync import ModbusSerialClient
from simple_x4_motor_wrapper import MotorWrapper
from typing import Literal


class MotorHardwareInterface:
    """
    Interface to six Fortune motors on the Hermes robot via Modbus RTU.
    Motors wiki: https://github.com/zavdimka/fortune-controls/wiki
    
    Each side has three positions: front.
    Configuration JSON files must be named like `config1L.json`, `config2R.json`, etc.
    """

    def __init__(
        self,
        serial_port: str = '/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0',
        config_dir: str = 'src/configs'
    ) -> None:
        """
        Initialize Modbus client and load motor wrappers from JSON configs.

        Args:
            serial_port: Path to serial device for Modbus RTU.
            config_dir: Directory containing motor config JSONs.
        """
        baudrate = 115200 # NOTE for Fortune motors baudrate is fixed
        self.client = ModbusSerialClient(
            method='rtu',
            port=serial_port,
            baudrate=baudrate,
            stopbits=1,
            bytesize=8,
            parity='N',
            timeout=0.02
        )
        if not self.client.connect():
            raise ConnectionError(f"Cannot connect to Modbus on {serial_port}@{baudrate}")
        print("Modbus is connected")

        for addr in range(1, 10):
            rr = self.client.read_holding_registers(0, 1, unit=addr)
            if rr and not hasattr(rr, 'isError') or not rr.isError():
                print(f"Got id {addr}: {rr}")
            else:
                print(f"Addr {addr}: {rr}")
        # asdsadsa
        # three slots per side: [front]
        self.left_motors = {'front': None}
        self.right_motors = {'front': None}

        map_idx_to_name = {1:"front"}
        connected = 0
        for cfg_path in glob.glob(os.path.join(config_dir, '*.json')):
            connected += 1
            name = os.path.splitext(os.path.basename(cfg_path))[0]  # e.g. 'config1L'
            is_left = name.endswith('L')
            position = map_idx_to_name[int(name[-2])]
            motor = MotorWrapper(self.client, config_path=cfg_path)
            (self.left_motors if is_left else self.right_motors)[position] = motor
            print(f"Motor {name} initialized")

        if connected != 2:
            raise ValueError(f"Expected 2 motors, found {connected}")

    def set_wheel_speeds(self, left_speed: float, right_speed: float) -> None:
        """
        Apply the same speed to all left motors and all right motors.

        Args:
            left_speed: Speed to set on each left motor.
            right_speed: Speed to set on each right motor.
        """
        for name, m in self.left_motors.items():
            # if m is None: continue
            # print(f"left {name} Error: {m.read_error()}")
            m.set_speed(left_speed)
        for name, m in self.right_motors.items():
            # if m is None: continue
            # print(f"right {name} Error: {m.read_error()}")
            m.set_speed(right_speed)

    def reset_motors(self, index = None):
        """
        Resets motors to restart after safety stop when no commands published 
        0, 1 - left front, right front
        """
        if index is None:  # recover all motors
            for motor in list(self.left_motors.values()) + list(self.right_motors.values()):
                if motor:
                    motor.reset_mode()
        motor = None    
        if index == 0: motor = self.left_motors["front"]
        if index == 1: motor = self.right_motors["front"]
        if motor:
            motor.reset_mode()

    def get_speed(
        self,
        side: Literal['left', 'right'],
        position: Literal['front'] = "front"
    ) -> float:
        """
        Read speed from a specific motor.

        Args:
            side: 'left' or 'right'.
            position: 'front'.
        """
        motors = self.left_motors if side == 'left' else self.right_motors
        motor = motors[position]
        if motor is None:
            raise ValueError(f"{side} {position} motor not initialized")
        return motor.read_speed()

    def get_angle(
        self,
        side: Literal['left', 'right'],
        position: Literal['front'] = "front"
    ) -> float:
        """
        Read angle from a specific motor.

        Args:
            side: 'left' or 'right'.
            position: 'front'.
        """
        motors = self.left_motors if side == 'left' else self.right_motors
        motor = motors[position]
        if motor is None:
            raise ValueError(f"{side} {position} motor not initialized")
        return motor.read_angle()



def main() -> None:
    import time
    interface = MotorHardwareInterface()

    while True:
        prev_time = time.time()
        for ang in range(-100, 100):
            t = time.time()
            ang_vel = ang / 25
            interface.set_wheel_speeds(ang_vel, -ang_vel)
            print(f"delay {1000*(t - prev_time):.3f} ms")
            prev_time = t

if __name__ == '__main__':
    main()
