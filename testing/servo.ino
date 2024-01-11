// Install the ServoEasing library through the Arduino Library Manager
#include <ServoEasing.hpp>

#define PanPot A0
#define TiltPot A1

ServoEasing servoPan; // Create a servo object
ServoEasing servoTilt;

float mapFloat(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
  return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow; 
}


void setup()
{
  Serial.begin(2000000); // Initialize serial communication at 9600 bps
  servoPan.attach(9, 80);    // Attach the servo to pin 9
  servoTilt.attach(10, 80);
}

void loop(){
  // Check if there's data available to read
  if (Serial.available() > 1)
  {
    // Read x and y coordinates from serial
    float targetX = Serial.parseFloat();
    float targetY = Serial.parseFloat();

    // Smoothly move the servos to the target position
    if (targetX > 0 && targetY > 0 && targetX < 1 &&targetY < 1)
      {
        targetX = int(mapFloat(targetX, 0, 1, 160, 0));
        targetY = int(mapFloat(targetY, 0, 1, 30, 160));

        servoPan.write(targetX);
        servoTilt.write(targetY);
      }
  }
}

void smoothMoveServo(ServoEasing& servo, int currentPos, int targetPos) {
  // Define the number of steps for smooth movement
  int steps = 1; // You can adjust this value based on your requirement
  // Use easing function for smooth movement
  servo.startEaseToD(targetPos, steps);
}