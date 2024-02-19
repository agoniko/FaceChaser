import serial

# Establish serial communication


class Arduino:
    def __init__(self, arduino_port="/dev/cu.usbmodem21301"):
        self.ser = serial.Serial(arduino_port, 2000000, timeout=1)

    def send_coordinates(self, x, y, z, IMG_SIZE=(1920, 1080), scaling_factor=1.0):
        """Send x,y,z coordinates in centimeters through serial communication"""
        dst_size = IMG_SIZE[0] * scaling_factor, IMG_SIZE[1] * scaling_factor
        x /= dst_size[0]
        y /= dst_size[1]
        x_max = 1920*z*25/(24*1080)
        y_max = 1080*z*25/(24*1080)
        #x = round(x, 3)
        #y = round(y, 3)
        x *= x_max
        y *= y_max
        x -= x_max/2
        y -= y_max/2
        data = f"{x},{y},{z}\n"
        self.ser.write(data.encode("utf-8"))

    def close(self):
        self.ser.close()
