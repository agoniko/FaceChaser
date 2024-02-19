#include <Servo.h>

#define BUFFLEN 10;

Servo servoPan;  // Create a servo object
Servo servoTilt;

float mapFloat(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
  return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow;
}

void setup() {
  Serial.begin(2000000);  // Initialize serial communication at 9600 bps
  servoPan.attach(9);     // Attach the servo to pin 9
  servoTilt.attach(10);
  servoPan.write(80);
  servoTilt.write(80);
}

void loop() {
  // Check if there's data available to read
  if (Serial.available() > 1) {
    // Read x and y coordinates from serial
    float targetX = Serial.parseFloat();
    float targetY = Serial.parseFloat();

    // Smoothly move the servos to the target position
    if (targetX > 0 && targetY > 0 && targetX < 1 && targetY < 1) {
      int pan = int(mapFloat(targetX, 0, 1, 0, 160));
      int tilt = int(mapFloat(targetY, 0, 1, 30, 160));

      servoPan.write(pan);
      servoTilt.write(tilt);
    }
  }
}