import serial
import numpy as np
from src.reference_frame_aware_vector import ReferenceFrame, ReferenceFrameAwareVector

# Establish serial communication


class Arduino:
    def __init__(self, serial_port, reference_frame):
        self.ser = serial.Serial(serial_port, 9600, timeout=1)
        self.reference_frame = reference_frame

    def set_computer_position(self, arduino_computer: np.ndarray):
        """Set computer position with respect to arduino reference system"""
        self.arduino_computer = arduino_computer

    def send_coordinates(self, target: ReferenceFrameAwareVector):
        """Send pan and tilt angles in degrees through serial communication"""
        target.to(self.reference_frame)

        pan = np.arccos(-target.vector[0] / np.sqrt(target.vector[0]**2 + target.vector[2]**2))
        tilt = np.arccos(-target.vector[1] / np.sqrt(target.vector[1]**2 + target.vector[2]**2))

        pan = np.rad2deg(pan)
        tilt = np.rad2deg(tilt)
        
        data = f"{int(pan)},{int(tilt)}\n"
        self.ser.write(data.encode("utf-8"))

    def close(self):
        self.ser.close()
