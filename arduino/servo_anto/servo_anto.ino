#include <Servo.h>

#define BUFFLEN 10;
#define PI 3.1415

float dx = 7.;
float dz = 25.+8.7;

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
    float targetX_computer = Serial.parseFloat();
    float targetY_computer = Serial.parseFloat();
    float targetZ_computer = Serial.parseFloat();

    float targetX_arduino = targetX_computer + dx;
    float targetZ_arduino = targetZ_computer - dz;

    float norm = sqrt(targetX_arduino*targetX_arduino + targetZ_arduino*targetZ_arduino);
    float versorX_target_arduino = targetX_arduino / norm;
    float versorZ_target_arduino = targetZ_arduino / norm;

    float pan_alpha_radians = acos(-versorX_target_arduino);
    
    // Clip to [PI/8, 7*PI/8] interval
    pan_alpha_radians = min(pan_alpha_radians, 7*PI/8);
    pan_alpha_radians = max(pan_alpha_radians, PI/8);

    int pan_alpha_degrees = int(mapFloat(pan_alpha_radians, 0., PI, 0., 180.));

    //int pan = int(mapFloat(alpha_degrees, 0, 1, 0, 160));
      //int tilt = int(mapFloat(targetY, 0, 1, 30, 160));

    servoPan.write(pan_alpha_degrees);
      //servoTilt.write(tilt);
  }
}