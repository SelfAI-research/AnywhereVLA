import time
import numpy as np
from dataclasses import dataclass
from motors_interface import MotorHardwareInterface

@dataclass
class RobotState:
    """
    Snapshot of the robots full state.
    """
    timestamp: float    # Timestamp of update [s]
    x: float            # X position in world frame [m]
    y: float            # Y position in world frame [m]
    theta: float        # Heading (yaw) [rad]
    linear_vel: float   # Forward speed [m/s]
    angular_vel: float  # Yaw rate [rad/s]
    left_ang: float     # Left front-wheel angle [rad]
    right_ang: float    # Right front-wheel angle [rad]

class RobotModel:
    """
    Kinematic model for the HermesBot differential drive.

    Uses the front wheel on each side to estimate chassis motion.
    Call `update()` each cycle, then `get_state()` to read out.
    """

    def __init__(
        self,
        wheel_radius: float = 0.085,
        wheel_base:   float = 0.455,
        max_wheel_speed: float = 20.0
    ) -> None:
        """
        Initialize kinematic parameters and motor interface.

        Args:
            wheel_radius:     Radius of each wheel [m].
            wheel_base:       Distance between left/right wheels [m].
            max_wheel_speed:  Max wheel angular speed [rad/s].
        """
        self.wheel_radius    = wheel_radius
        self.wheel_base      = wheel_base
        self.max_wheel_speed = max_wheel_speed

        # Motor interface
        self.interface = MotorHardwareInterface()

        # Initial sensor readings & state
        left_ang  = self.interface.get_angle('left',  'front')
        right_ang = self.interface.get_angle('right', 'front')
        now       = time.perf_counter()

        # Initialize state dataclass
        self.state = RobotState(
            timestamp=now,
            x=0.0,
            y=0.0,
            theta=0.0,
            linear_vel=0.0,
            angular_vel=0.0,
            left_ang=left_ang,
            right_ang=right_ang,
        )

    def motors_recovery(self, index = None) -> None:
        self.interface.reset_motors(index)

    def set_velocity(self, linear_vel: float, angular_vel: float) -> None:
        """
        Command the robot to move with given linear & angular velocity.

        Converts to wheel speeds, clamps them, and sends to motors.
        """
        ω_l, ω_r = self.inverse_kinematics(linear_vel, angular_vel)
        ω_l = np.clip(ω_l, -self.max_wheel_speed, self.max_wheel_speed)
        ω_r = np.clip(ω_r, -self.max_wheel_speed, self.max_wheel_speed)
        self.interface.set_wheel_speeds(ω_l, ω_r)

    def update(self) -> None:
        """
        Read sensors, compute velocities, and update pose.

        Should be called exactly once per control loop iteration.
        """
        now = time.perf_counter()
        dt  = now - self.state.timestamp
        if dt <= 0.0:
            return

        # 1) Read new angles
        left_ang  = self.interface.get_angle('left',  'front')
        right_ang = self.interface.get_angle('right', 'front')

        # 2) Compute wheel angular velocities [rad/s]
        dϕ_l = left_ang  - self.state.left_ang
        dϕ_r = right_ang - self.state.right_ang
        ω_l  = dϕ_l / dt
        ω_r  = dϕ_r / dt

        # 3) Forward kinematics → chassis velocities
        v, ω = self.forward_kinematics(ω_l, ω_r)
        self.state.linear_vel  = v
        self.state.angular_vel = ω

        # 4) Euler‐integrate pose
        self.state.x     += v * np.cos(self.state.theta) * dt
        self.state.y     += v * np.sin(self.state.theta) * dt
        self.state.theta += ω * dt

        # 5) Save new readings
        self.state.left_ang  = left_ang
        self.state.right_ang = right_ang
        self.state.timestamp = now

    def get_state(self) -> RobotState:
        """
        Get the latest pose and velocity estimate.
        """
        return self.state

    def inverse_kinematics(
        self,
        linear_vel: float,
        angular_vel: float
    ) -> tuple[float, float]:
        """
        Compute wheel speeds from chassis commands.

        Args:
            linear_vel:  Forward speed [m/s].
            angular_vel: Yaw rate [rad/s].

        Returns:
            (omega_left, omega_right): wheel angular speeds [rad/s].
        """
        v_l = linear_vel - (self.wheel_base / 2.0) * angular_vel
        v_r = linear_vel + (self.wheel_base / 2.0) * angular_vel
        return v_l / self.wheel_radius, v_r / self.wheel_radius

    def forward_kinematics(
        self,
        omega_left:  float,
        omega_right: float
    ) -> tuple[float, float]:
        """
        Convert wheel speeds back into chassis motion.

        Args:
            omega_left:  Left wheel speed [rad/s].
            omega_right: Right wheel speed [rad/s].

        Returns:
            (linear_vel, angular_vel):
                linear_vel  [m/s], angular_vel [rad/s].
        """
        v_l = omega_left  * self.wheel_radius
        v_r = omega_right * self.wheel_radius
        linear  = (v_l + v_r) / 2.0
        angular = (v_r - v_l) / self.wheel_base
        return linear, angular
