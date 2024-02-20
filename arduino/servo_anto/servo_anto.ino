#include <Servo.h>
#define PI 3.1415

float mapFloat(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
  return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow;
}

typedef struct{
  float x;
  float y;
  float z;
} Vector;

// Objects coordinates
// variabale names notation: <object>_<reference system>
Vector computer_arduino[2];
Vector target_arduino;
Vector target_computer;

// Pan, Tilt angles
float pan_angle_radians;
float tilt_angle_radians;

float pan_angle_degrees;
float tilt_angle_degrees;

// Servo objects
Servo servoPan[2];
Servo servoTilt[2];

void setup() {
  Serial.begin(2000000);

  // Servo 0
  servoPan[0].attach(9);
  servoTilt[0].attach(10);
  servoPan[0].write(80);
  servoTilt[0].write(80);

  // Servo 1
  servoPan[1].attach(7);
  servoTilt[1].attach(8);
  servoPan[1].write(80);
  servoTilt[1].write(80);

  // camera position with respect to servo motors
  computer_arduino[0].x = 7.;
  computer_arduino[0].y = 22.;
  computer_arduino[0].z = -(25. + 8.7);

  computer_arduino[1].x = -7.;
  computer_arduino[1].y = 22.;
  computer_arduino[1].z = -(25. + 8.7);
}

void loop() {
  // Check if there's data available to read
  int idx;
  if (Serial.available() > 1) {
    // Read arduino index
    idx = Serial.parseInt();

    // Read x,y,z coordinates from serial
    target_computer.x = Serial.parseFloat();
    target_computer.y = Serial.parseFloat();
    target_computer.z = Serial.parseFloat();

    // Compute relative position to arduino
    target_arduino.x =   target_computer.x + computer_arduino[idx].x;
    target_arduino.y = - target_computer.y + computer_arduino[idx].y;
    target_arduino.z =   target_computer.z + computer_arduino[idx].z;

    pan_angle_radians = acos(
      -target_arduino.x /
      sqrt(target_arduino.x*target_arduino.x + target_arduino.z * target_arduino.z)
    );

    tilt_angle_radians = acos(
      -target_arduino.y /
      sqrt(target_arduino.y*target_arduino.y + target_arduino.z * target_arduino.z)
    );
    
    // Clip to [PI/8, 7*PI/8] interval
    pan_angle_radians = min(pan_angle_radians, 7*PI/8);
    pan_angle_radians = max(pan_angle_radians, PI/8);

    tilt_angle_radians = min(tilt_angle_radians, 7*PI/8);
    tilt_angle_radians = max(tilt_angle_radians, PI/8);

    // Map to degrees
    pan_angle_degrees = mapFloat(pan_angle_radians, 0., PI, 0., 180.);
    tilt_angle_degrees = mapFloat(tilt_angle_radians, 0., PI, 0., 180.);

    servoPan[idx].write(int(pan_angle_degrees));
    servoTilt[idx].write(int(tilt_angle_degrees));
  }
}