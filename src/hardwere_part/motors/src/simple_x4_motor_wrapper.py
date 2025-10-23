import os
import time
import hjson
import numpy as np
from pymodbus.client.sync     import ModbusSerialClient
from pymodbus.payload         import BinaryPayloadBuilder, BinaryPayloadDecoder
from pymodbus.constants       import Endian
from pymodbus.pdu             import ExceptionResponse

class MotorWrapper:
    # Modbus register addresses
    SETPOINT_REG   = 3     # where we write our target
    FEEDBACK_REG   = 69    # where the controller reports actual speed
    ANGLE_REG      = 67    # two registers (32-bit) for encoder count
    ERROR_REG      = 29
    MODE_REG       = 0
    VMIN_REG       = 9
    ILIMIT_REG     = 10
    TEMP_REG       = 11
    TIMEOUT_REG    = 18
    PWM_LIMIT_REG  = 21
    PWM_INC_REG    = 22
    SPD_PID_P_REG  = 13
    SPD_PID_I_REG  = 12

    # Mode values
    MODE_SPEED = 2

    def __init__(self, client: ModbusSerialClient, config_path: str):
        """
        client:    an open pymodbus ModbusSerialClient
        config_path: path to your HJSON with motor settings
        """
        self.client = client

        # Load the HJSON config (allows // comments)
        cfg = hjson.load(open(config_path))

        self.id            = cfg.get('id', 1)
        self.reverse       = bool(cfg.get('Reverse', 0))
        self.ticks_per_rev = float(cfg.get('TicksPerRev', 3840.0))
        self.gear_ratio    = float(cfg.get('GearRatio', 22.0))

        # 3) apply the hardware limits & PID gains you care about
        if 'V_min' in cfg:
            client.write_register(self.VMIN_REG, int(cfg['V_min'] * 1000), unit=self.id)
        if 'I_limit' in cfg:
            client.write_register(self.ILIMIT_REG, int(cfg['I_limit'] * 1000), unit=self.id)
        if 'TempShutDown' in cfg:
            client.write_register(self.TEMP_REG, int(cfg['TempShutDown']), unit=self.id)
        if 'TimeOut' in cfg:
            # controller expects timeout in units of ~40ms
            client.write_register(self.TIMEOUT_REG, int(cfg['TimeOut'] / 40), unit=self.id)

        if 'PWM_Limit' in cfg:
            client.write_register(self.PWM_LIMIT_REG, int(cfg['PWM_Limit']), unit=self.id)
        if 'PWM_inc_limit' in cfg:
            client.write_register(self.PWM_INC_REG, int(cfg['PWM_inc_limit']), unit=self.id)

        if 'Speed_PID_P' in cfg:
            client.write_register(self.SPD_PID_P_REG, int(cfg['Speed_PID_P']), unit=self.id)
        if 'Speed_PID_I' in cfg:
            client.write_register(self.SPD_PID_I_REG, int(cfg['Speed_PID_I']), unit=self.id)
        self.reset_mode()

    def reset_mode(self):
        """
        Clear any latched faults (including the timeout watchdog) by
        writing 0 into the error register.
        """
        # Clear the latched fault:
        self.client.write_register(self.ERROR_REG, 0, unit=self.id)
        time.sleep(0.02)   # give the controller a moment

        # Re-enter speed mode so it will accept new setpoints:
        self.client.write_register(self.MODE_REG, self.MODE_SPEED, unit=self.id)
        time.sleep(0.02)

        self.set_speed(0)

    def set_speed(self, rad_s: float):
        """
        Set wheel speed in radians/sec.
        Converts to controller ticks/sec using ticks_per_rev.
        """
        modified_rad_s = rad_s / self.gear_ratio

        ticks_per_s = self.ticks_per_rev / (2 * np.pi) * modified_rad_s
        if self.reverse:
            ticks_per_s = -ticks_per_s

        builder = BinaryPayloadBuilder(
            byteorder=Endian.Big,
            wordorder=Endian.Little
        )
        builder.add_16bit_int(int(round(ticks_per_s)))
        reg = builder.to_registers()[0]
        self.client.write_register(self.SETPOINT_REG, reg, unit=self.id)

    def read_error(self) -> int:
        """
        Read the controller’s internal error code.
        Zero usually means “no fault.”
        """
        rr = self.client.read_holding_registers(self.ERROR_REG, 1, unit=self.id)
        if isinstance(rr, ExceptionResponse) or not hasattr(rr, 'registers'):
            print(f"Failed to read ERROR_REG: {rr}")
        return rr

    def read_speed(self) -> float:
        """
        Read the actual wheel speed in radians/sec.
        Reads the feedback register (raw ticks/sec) and converts to rad/sec.
        """
        rr = self.client.read_holding_registers(self.FEEDBACK_REG, 1, unit=self.id)
        regs = rr.registers if hasattr(rr, 'registers') else [0]
        decoder = BinaryPayloadDecoder.fromRegisters(
            regs,
            byteorder=Endian.Big,
            wordorder=Endian.Little
        )
        raw = decoder.decode_16bit_int()   # now raw is in –32768…+32767
        if self.reverse:
            raw = -raw

        modified_raw = self.gear_ratio * raw
        return (2 * np.pi) / self.ticks_per_rev * modified_raw

    def read_angle(self) -> int:
        """
        Read the wheel angle in radian
        """
        rr = self.client.read_holding_registers(self.ANGLE_REG, 2, unit=self.id)
        regs = rr.registers if hasattr(rr, 'registers') else [0, 0]
        decoder = BinaryPayloadDecoder.fromRegisters(
            regs,
            byteorder=Endian.Big,
            wordorder=Endian.Little
        )
        encoder_ticks = decoder.decode_32bit_int()
        if self.reverse:
            encoder_ticks = -encoder_ticks
        return (2 * np.pi) / self.ticks_per_rev * encoder_ticks
