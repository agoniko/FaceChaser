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
float targetX_computer;
float targetY_computer;
float targetZ_computer;

float targetX_arduino;
float targetY_arduino;
float targetZ_arduino;

float versorX_target_arduino;
float versorY_target_arduino;
float versorZ_target_arduino;

float pan_alpha_radians;
float tilt_alpha_radians;

int pan_alpha_degrees;
int tilt_alpha_degrees;

float norm;



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
    targetX_computer = Serial.parseFloat();
    targetY_computer = Serial.parseFloat();
    targetZ_computer = Serial.parseFloat();

    targetX_arduino = targetX_computer + dx;
    targetZ_arduino = targetZ_computer - dz;

    norm = sqrt(targetX_arduino*targetX_arduino + targetZ_arduino*targetZ_arduino);
    versorX_target_arduino = targetX_arduino / norm;
    versorZ_target_arduino = targetZ_arduino / norm;

    pan_alpha_radians = acos(-versorX_target_arduino);
    
    // Clip to [PI/8, 7*PI/8] interval
    pan_alpha_radians = min(pan_alpha_radians, 7*PI/8);
    pan_alpha_radians = max(pan_alpha_radians, PI/8);

    pan_alpha_degrees = int(mapFloat(pan_alpha_radians, 0., PI, 0., 180.));

    //int pan = int(mapFloat(alpha_degrees, 0, 1, 0, 160));
      //int tilt = int(mapFloat(targetY, 0, 1, 30, 160));

    servoPan.write(pan_alpha_degrees);
    servoTilt.write(pan_alpha_degrees);
      //servoTilt.write(tilt);
  }
}