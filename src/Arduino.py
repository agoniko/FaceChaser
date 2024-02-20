import serial
import numpy as np

# Establish serial communication


class Arduino:
    def __init__(self, arduino_port="/dev/cu.usbmodem21301"):
        self.ser = serial.Serial(arduino_port, 9600, timeout=1)

        # Assume computer and arduino positions coincide
        self.arduino_computer = np.array([0., 0., 0.])

    def set_computer_position(self, arduino_computer: np.ndarray):
        """Set computer position with respect to arduino reference system"""
        self.arduino_computer = arduino_computer

    def send_coordinates(self, x, y, z, IMG_SIZE=(1920, 1080), scaling_factor=1.0):
        """Send pan and tilt angles in degrees through serial communication"""
        dst_size = IMG_SIZE[0] * scaling_factor, IMG_SIZE[1] * scaling_factor
        x /= dst_size[0]
        y /= dst_size[1]

        x_max = 1920*z*25/(24*1080)
        y_max = 1080*z*25/(24*1080)
        
        x *= x_max
        y *= y_max

        x -= x_max/2
        y -= y_max/2

        # Given two points in space P1, P2
        # <P1>_<P2> is the vector P2-P1
        computer_target = np.array([x, y, z])
        arduino_target = self.arduino_computer + computer_target

        pan = np.arccos(-arduino_target[0] / np.sqrt(arduino_target[0]**2 + arduino_target[2]**2))
        tilt = np.arccos(-arduino_target[1] / np.sqrt(arduino_target[1]**2 + arduino_target[2]**2))

        pan = np.rad2deg(pan)
        tilt = np.rad2deg(tilt)
        
        data = f"{int(pan)},{int(tilt)}\n"
        self.ser.write(data.encode("utf-8"))

    def close(self):
        self.ser.close()
