#include <Servo.h>

// Create servo objects for horizontal and vertical servos
Servo horizontalServo;
Servo verticalServo;

// Define the analog pins for joystick inputs (swapped)
const int VRxPin = A1;  // Joystick horizontal axis
const int VRyPin = A0;  // Joystick vertical axis

// Define the digital pins for the servos
const int horizontalServoPin = 9;
const int verticalServoPin = 10;

// Define the digital pins for the laser and joystick switch
const int laserPin = 8;  // Laser control pin
const int joystickSwPin = 7;  // Joystick switch pin

// Variables to store the current position of the servos
int horizontalPosition = 90;
int verticalPosition = 90;

// Sensitivity adjustments
const int speedFactor = 15;  // Decreased for more responsiveness
const int deadzone = 10;     // Increased deadzone for less sensitivity

// Variables for laser state and debounce logic
bool laserState = false;  // Laser initially off
bool lastButtonState = HIGH;  // Previous button state
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;  // Debounce time in milliseconds

void setup() {
  // Attach the servos to their pins
  horizontalServo.attach(horizontalServoPin);
  verticalServo.attach(verticalServoPin);

  // Initialize the servos to center position
  horizontalServo.write(horizontalPosition);
  verticalServo.write(verticalPosition);

  // Set laser control pin as output and initialize it to LOW
  pinMode(laserPin, OUTPUT);
  digitalWrite(laserPin, LOW);

  // Set joystick switch pin as input with pull-up resistor
  pinMode(joystickSwPin, INPUT_PULLUP);

  // Begin serial communication for debugging (optional)
  Serial.begin(9600);
}

void loop() {
  // Read the joystick values
  int xValue = analogRead(VRxPin);
  int yValue = analogRead(VRyPin);

  // Map joystick values to speed range (-15 to +15) for more responsiveness
  int horizontalSpeed = map(xValue, 0, 1023, -15, 15);
  int verticalSpeed = map(yValue, 0, 1023, -15, 15);

  // Apply deadzone filtering
  if (abs(horizontalSpeed) < deadzone) horizontalSpeed = 0;
  if (abs(verticalSpeed) < deadzone) verticalSpeed = 0;

  // Update the servo positions
  horizontalPosition = constrain(horizontalPosition + horizontalSpeed / speedFactor, 0, 180);
  verticalPosition = constrain(verticalPosition + verticalSpeed / speedFactor, 0, 180);

  // Write the updated positions to the servos
  horizontalServo.write(horizontalPosition);
  verticalServo.write(verticalPosition);

  // Read the joystick switch state with debounce logic
  bool currentButtonState = digitalRead(joystickSwPin);
  
  if (currentButtonState != lastButtonState) {
    lastDebounceTime = millis();  // Reset the debounce timer
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    // If the button state has changed and is LOW (pressed), toggle the laser
    if (currentButtonState == LOW) {
      laserState = !laserState;  // Toggle the laser state
      digitalWrite(laserPin, laserState ? HIGH : LOW);
    }
  }

  // Save the current button state for the next iteration
  lastButtonState = currentButtonState;

  // Debugging output (optional)
  Serial.print("Horizontal Position: ");
  Serial.print(horizontalPosition);
  Serial.print("  Vertical Position: ");
  Serial.print(verticalPosition);
  Serial.print("  Laser State: ");
  Serial.println(laserState ? "ON" : "OFF");

  // Small delay to reduce jitter
  delay(20);
}
