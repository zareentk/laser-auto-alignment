#include <Servo.h>

Servo xServo;
Servo yServo;

const unsigned long timeout = 10000; // Timeout in milliseconds (1 second)
unsigned long lastDataTime = 0; // Tracks the last time data was received

void setup() {
  xServo.attach(9);  // Attach the servo to pin 9
  yServo.attach(8);
  xServo.write(120);  // Set initial position to center
  yServo.write(25);
  Serial.begin(115200);
  Serial.setTimeout(1);
}

void processPythonInput(int value1, int value2) {

  //Serial.println(yCoord);
  // Update servo position
  xServo.write(value1);
  yServo.write(value2);
  //String Test = String(targetPositionX) + String(targetPositionY);
  //Serial.println(Test);


  lastDataTime = millis(); // Update last received time
}


void loop() {
  // Check if data is available
  if (Serial.available() > 0) {
    delay(1);
    String inputString = Serial.readStringUntil('\n');  // Read full line
    int value1, value2;
    if (sscanf(inputString.c_str(), "%d,%d", &value1, &value2) == 2) {  // Parse both values
    processPythonInput(value1,value2);
  }

  // Check if timeout has elapsed (no data received for `timeout` ms)
  if (millis() - lastDataTime > timeout) {
    xServo.write(120); // Move servo back to center
    yServo.write(25);
  }
}
}