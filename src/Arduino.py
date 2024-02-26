import serial
from typing import List

import numpy as np

from src.refsys.system import ReferenceSystem
from src.refsys.vector import Vector


class Arduino:
    def __init__(self, serial_port, ref_sys_list: List[ReferenceSystem]):
        if serial_port != "virtual":
            self.ser = serial.Serial(serial_port, 9600, timeout=1)
            self.virtual = False
        else:
            self.virtual = True
        self.ref_sys_list = ref_sys_list

    def send_coordinates(self, target: Vector):
        """Send pan and tilt angles in degrees through serial communication"""
        target.to(self.ref_sys_list[-1])
        target.detach()

        pan = np.arccos(-target.array[0] / np.sqrt(target.array[0]**2 + target.array[2]**2))
        tilt = np.arccos(-target.array[1] / np.sqrt(target.array[1]**2 + target.array[2]**2))

        pan = np.rad2deg(pan)
        tilt = np.rad2deg(tilt)
        
        data = f"{int(pan)},{int(tilt)}\n"
        if not self.virtual:
            self.ser.write(data.encode("utf-8"))

    def close(self):
        if not self.virtual:
            self.ser.close()
