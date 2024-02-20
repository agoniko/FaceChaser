#include <Servo.h>

#define BUFFLEN 10;

Servo servoPan;  // Create a servo object
Servo servoTilt;

float mapFloat(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
  return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow;
}
int xmax;
int ymax;

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
    // Read x and y coordinates from serial
    float targetX = Serial.parseFloat();
    float targetY = Serial.parseFloat();
    float targetZ = Serial.parseFloat();
    
    xmax = 1920*targetZ*25/(24*1080);
    ymax = 1080*targetZ*25/(24*1080);

    targetX += xmax/2;
    targetY += ymax/2;
    targetX /= max(xmax, 1e-5);
    targetY /= max(ymax, 1e-5);

    // Smoothly move the servos to the target position
    if (targetX > 0 && targetY > 0 && targetX < 1 && targetY < 1) {
      int pan = int(mapFloat(targetX, 0, 1, 0, 180));
      int tilt = int(mapFloat(targetY, 0, 1, 0, 180));

      servoPan.write(pan);
      servoTilt.write(tilt);
    }
  }
}