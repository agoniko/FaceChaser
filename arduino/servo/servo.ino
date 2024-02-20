#include <Servo.h>

Servo servoPan;  // Create a servo object
Servo servoTilt;

int pan;
int tilt;

void setup() {
  Serial.begin(9600);  // Initialize serial communication at 9600 bps
  servoPan.attach(9);     // Attach the servo to pin 9
  servoTilt.attach(10);
  servoPan.write(90);
  servoTilt.write(90);
}

void loop() {
  // Check if there's data available to read
  if (Serial.available() > 0) {
    pan = Serial.parseInt();
    tilt = Serial.parseInt();

    servoPan.write(pan);
    servoTilt.write(tilt);
  }
}