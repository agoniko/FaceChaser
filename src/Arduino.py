import serial

# Establish serial communication


class Arduino:
    def __init__(self, IMG_SIZE: tuple, arduino_port="/dev/cu.usbmodem21301"):
        #self.ser = serial.Serial(arduino_port, 2000000, timeout=1)
        self.w = IMG_SIZE[0]
        self.h = IMG_SIZE[1]

    def send_coordinates(self, x, y):
        x /= self.w
        y /= self.h
        x = round(x, 3)
        y = round(y, 3)
        data = f"{x},{y}\n"
        #self.ser.write(data.encode("utf-8"))

    def close(self):
        self.ser.close()
