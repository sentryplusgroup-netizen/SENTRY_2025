#include <AccelStepper.h>

// ==================== ANGULAR COMPENSATION CALIBRATION ====================
// Base offsets at home position (0,0) - your current working values
// NOTE: Laser is ABOVE camera, perfectly aligned horizontally
#define BASE_X_OFFSET -10    // Static horizontal offset if laser drifts left/right at center
#define BASE_Y_OFFSET 40   // Main vertical offset - angular compensation applied here

// Motor configuration for angular compensation (both axes)
const int stepsPerRevolution = 200;
const int microstepping = 2;  // Half-stepping = 2x steps per revolution (400 steps/rev)
const float STEPS_PER_DEGREE_Y = (stepsPerRevolution * microstepping) / 360.0;  // ~1.11 steps/degree
const float STEPS_PER_DEGREE_X = (stepsPerRevolution * microstepping * 10) / 360.0;  // ~11.11 steps/degree (with 10:1 gear ratio)

// ==================== LASER ====================
String serialBuffer = "";
const int Laser = 8; // Laser connected to pin 8 on the arduino
bool laser_state = false; 
unsigned long lastCoordTime = 0;
const unsigned long LASER_TIMEOUT = 5000;  // 5 seconds - prevents premature laser shutoff during tracking

/*******Declare variables for X rotation stepper motor*******/

const int LimitX1pin = A5; // Limit switch one connected to pin A5
const int LimitX2pin = A0; // Limit switch two connected to pin A0
const int MotorX_ENApin = 6; // Set the enableA for coil 1 on pin 6
const int MotorX_ENBpin = 5; // Set the enableB for coil 2 on pin 5
bool Home_pos_Xmotor = false;

/*******Declare variables for Y rotation stepper motor*******/

const int LimitY1pin = 12; // Limit switch one connected to pin 12
const int LimitY2pin = 13; // Limit switch two connected to pin 13
const int MotorY_ENApin = 9; // Set the enableA for coil 1 on pin 9
const int MotorY_ENBpin = 10; // Set the enableB for coil 2 on pin 10
bool Home_pos_Ymotor = false;

// Stepper motors setup
int Fullrotation_steps = 200; // Steps for a full 360 degrees rotation of the motors

// AccelStepper objects (4-wire L298N style drivers in HALF-STEP mode)
AccelStepper stepperX(AccelStepper::HALF4WIRE, 7, 4, 3, 2);
AccelStepper stepperY(AccelStepper::HALF4WIRE, A1, A2, A3, A4);

// Store motor steps taken to keep track of position
long currentStepPos_X = 0; 
long currentStepPos_Y = 0;

// Pixel conversions
const int centerX = 160; // Center of the camera frame - 320/2=160 for X axis
const int centerY = 160; // Center of the camera frame - 320/2=160 for Y axis

// IMPORTANT: X-axis has 10:1 gear reduction, Y-axis is direct drive
// Half-stepping doubles the steps, so double the values
const float stepsPerPixel_X = 0.1736 * 10 * 2; // X-axis: 10:1 reduction × half-stepping = 3.472 steps/pixel
const float stepsPerPixel_Y = 0.1736 * 2;      // Y-axis: direct drive × half-stepping = 0.3472 steps/pixel

// Deadband: ignore tiny movements to prevent overshoot oscillation
const int DEADBAND_STEPS_X = 5;  // ignore movements smaller than 5 steps on X (~1.4 pixels)
const int DEADBAND_STEPS_Y = 2;   // ignore movements smaller than 2 steps on Y (~5.8 pixels)

// ----- Hybrid auto-disable timing (per-axis) -----
bool motorX_enabled = false;
bool motorY_enabled = false;
unsigned long lastMoveTimeX = 0;
unsigned long lastMoveTimeY = 0;
const unsigned long MOTOR_DISABLE_DELAY = 300; // ms after finished moving before disabling

/*******Enable/disable functions for X rotation stepper motor*******/

void enableMotorX() {
  if (!motorX_enabled) {
    int PWMvalue = 78;
    analogWrite(MotorX_ENApin, PWMvalue);
    analogWrite(MotorX_ENBpin, PWMvalue);
    motorX_enabled = true;
  }
}

void disableMotorX() {
  if (motorX_enabled) {
    analogWrite(MotorX_ENApin, 0);
    analogWrite(MotorX_ENBpin, 0);
    motorX_enabled = false;
  }
}

/*******Enable/disable functions for Y rotation stepper motor*******/

void enableMotorY() {
  if (!motorY_enabled) {
    int PWMvalue = 78;
    analogWrite(MotorY_ENApin, PWMvalue);
    analogWrite(MotorY_ENBpin, PWMvalue);
    motorY_enabled = true;
  }
}

void disableMotorY() {
  if (motorY_enabled) {
    analogWrite(MotorY_ENApin, 0);
    analogWrite(MotorY_ENBpin, 0);
    motorY_enabled = false;
  }
}

/*******Homing function for X rotation stepper motor*******/

void HomeMotorX() {
  enableMotorX();
  
  // Check if it is already at home position
  if (digitalRead(LimitX1pin) == LOW && digitalRead(LimitX2pin) == LOW) {
    Home_pos_Xmotor = true;
    stepperX.setCurrentPosition(0); // define this as 0
    Serial.println("Already at HOME position on startup (X)");
    return;
  } 
  
  // Not already home -> do a scan: +90° then -180°
  Serial.println("Homing X with 90° + 180° scan...");

  // ----- FIRST SCAN: +90° (~500 steps) -----
  stepperX.move(550);  // positive direction, adjust sign if wrong
  while (stepperX.distanceToGo() != 0) {
    stepperX.run();

    bool limitX1 = digitalRead(LimitX1pin);
    bool limitX2 = digitalRead(LimitX2pin);

    // If both limits pressed, we are at HOME
    if (limitX1 == LOW && limitX2 == LOW) {
      Home_pos_Xmotor = true;
      stepperX.setCurrentPosition(0);
      Serial.println("Stepper Motor X at HOME position (found during +90° scan)");
      delay(200);
      return;
    }
  }

  // ----- SECOND SCAN: -180° (~1000 steps other way) -----
  stepperX.move(-1000);
  while (stepperX.distanceToGo() != 0) {
    stepperX.run();

    bool limitX1 = digitalRead(LimitX1pin);
    bool limitX2 = digitalRead(LimitX2pin);

    if (limitX1 == LOW && limitX2 == LOW) {
      Home_pos_Xmotor = true;
      stepperX.setCurrentPosition(0);
      Serial.println("Stepper Motor X at HOME position (found during -180° scan)");
      delay(200);
      return;
    }
  }

  // If we get here, something is wrong (switches not hit)
  Serial.println("ERROR: Could not find HOME for X after scan!");
}

/*******Homing function for Y rotation stepper motor*******/

void HomeMotorY() {
  enableMotorY();
  
  // Check if it is already at home position
  if (digitalRead(LimitY1pin) == LOW && digitalRead(LimitY2pin) == LOW) {
    Home_pos_Ymotor = true;
    stepperY.setCurrentPosition(0); // define this as 0
    Serial.println("Already at HOME position on startup (Y)");
    return;
  } 
  
  // Not already home -> do a scan: 50° one way, then 100° the other way
  Serial.println("Homing Y with +50° & -100° scan...");

  // ----- FIRST SCAN: try 50° (~28 steps) -----
  stepperY.move(28);
  while (stepperY.distanceToGo() != 0) {
    stepperY.run();

    bool limitY1 = digitalRead(LimitY1pin);
    bool limitY2 = digitalRead(LimitY2pin);

    if (limitY1 == LOW && limitY2 == LOW) {
      Home_pos_Ymotor = true;
      stepperY.setCurrentPosition(0);
      Serial.println("Stepper Motor Y at HOME position (found during +50° scan)");
      delay(200);
      return;
    }
  }

  // ----- SECOND SCAN: +100° (~56 steps the other way) -----
  stepperY.move(-56);
  while (stepperY.distanceToGo() != 0) {
    stepperY.run();

    bool limitY1 = digitalRead(LimitY1pin);
    bool limitY2 = digitalRead(LimitY2pin);

    if (limitY1 == LOW && limitY2 == LOW) {
      Home_pos_Ymotor = true;
      stepperY.setCurrentPosition(0);
      Serial.println("Stepper Motor Y at HOME position (found during -100° scan)");
      delay(200);
      return;
    }
  }

  // If we get here, something is wrong (switches not hit)
  Serial.println("ERROR: Could not find HOME for Y after scan!");
}

// ==================== ANGULAR COMPENSATION FUNCTION ====================
void calculateCorrectedOffsets(float targetAngleX, float targetAngleY, float &correctedX, float &correctedY) {
  /*
   * Apply 2-axis trigonometric correction for laser offset
   * 
   * Since laser is ABOVE camera by 50 pixels:
   * - When turret PANS left/right, the vertical offset causes horizontal parallax
   * - When turret TILTS up/down, the vertical offset changes apparent position
   * 
   * Math:
   * - correctedX = vertical_offset × sin(pan_angle)  ← horizontal shift due to pan
   * - correctedY = vertical_offset × cos(tilt_angle) ← vertical shift due to tilt
   */
  
  // Convert degrees to radians
  float angleX_rad = targetAngleX * PI / 180.0;
  float angleY_rad = targetAngleY * PI / 180.0;
  
  // Apply trigonometric scaling
  correctedX = BASE_Y_OFFSET * sin(angleX_rad);  // Horizontal correction from pan angle
  correctedY = BASE_Y_OFFSET * cos(angleY_rad);  // Vertical correction from tilt angle
}

void setup() {
  Serial.begin(9600);
  Serial.println("Sentry Turret Starting (2-Axis Angular Compensation Mode)...");
  
  // Laser setup
  pinMode(Laser, OUTPUT);
  digitalWrite(Laser, LOW); // Laser off initially

  // X-axis limits and motor driver enable pins
  pinMode(LimitX1pin, INPUT_PULLUP);
  pinMode(LimitX2pin, INPUT_PULLUP);
  pinMode(MotorX_ENApin, OUTPUT);
  pinMode(MotorX_ENBpin, OUTPUT);

  // Y-axis limits and motor driver enable pins
  pinMode(LimitY1pin, INPUT_PULLUP);
  pinMode(LimitY2pin, INPUT_PULLUP);
  pinMode(MotorY_ENApin, OUTPUT);
  pinMode(MotorY_ENBpin, OUTPUT);

  // Set stepper motor parameters (AccelStepper uses max speed + acceleration)
  // Balanced for accuracy and responsiveness
  stepperX.setMaxSpeed(600);
  stepperX.setAcceleration(150);   // balanced for smooth tracking without overshoot
  stepperY.setMaxSpeed(300);
  stepperY.setAcceleration(60);   // lower for Y-axis precision
  
  Serial.println("Homing the stepper motor");
  HomeMotorX();
  HomeMotorY();
  Serial.println("Homing complete");

  disableMotorX();
  disableMotorY();
  
  Serial.println("System Ready - 2-Axis Angular Compensation Active");
  Serial.print("Steps per degree - X: ");
  Serial.print(STEPS_PER_DEGREE_X);
  Serial.print(" | Y: ");
  Serial.println(STEPS_PER_DEGREE_Y);
  
  delay(500);
}

void loop() {
  // AccelStepper needs this called as often as possible for smooth motion
  stepperX.run();
  stepperY.run();

  while (Serial.available()) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n') {
      // We received a full message
      String message = serialBuffer;
      serialBuffer = "";

      message.trim();

      // Handle laser ON
      if (message == "Deer detected") {
        if (!laser_state) {
          digitalWrite(Laser, HIGH);
          laser_state = true;
          Serial.println("Laser ON");
        }
        lastCoordTime = millis();
      }

      // Handle laser OFF
      else if (message == "No deer") {
        if (laser_state) {
          digitalWrite(Laser, LOW);
          laser_state = false;
          Serial.println("Laser OFF");
        }
        lastCoordTime = millis();
      }

      // Handle X,Y coordinates with angular compensation
      else {
        int commaIndex = message.indexOf(',');
        if (commaIndex > 0) {
          int Xcoord = message.substring(0, commaIndex).toInt();
          int Ycoord = message.substring(commaIndex + 1).toInt();

          lastCoordTime = millis();

          // ========== FULL 2-AXIS ANGULAR COMPENSATION ==========
          // Calculate target angles for both axes FIRST
          int rawDeltaX = Xcoord - centerX;
          int rawDeltaY = centerY - Ycoord;
          
          long rawTargetStepsX = round(rawDeltaX * stepsPerPixel_X);
          long rawTargetStepsY = round(rawDeltaY * stepsPerPixel_Y);
          
          float targetAngleX = rawTargetStepsX / STEPS_PER_DEGREE_X;  // Pan angle
          float targetAngleY = rawTargetStepsY / STEPS_PER_DEGREE_Y;  // Tilt angle
          
          // Calculate corrected offsets based on both pan and tilt angles
          float correctedX, correctedY;
          calculateCorrectedOffsets(targetAngleX, targetAngleY, correctedX, correctedY);
          
          // Apply corrections (note: BASE_X_OFFSET is 0, but kept for clarity)
          int deltaX = Xcoord - centerX + BASE_X_OFFSET - (int)correctedX;
          int deltaY = centerY - Ycoord - (int)correctedY;
          // ==========================================

          long targetStepPos_X = round(deltaX * stepsPerPixel_X);
          long targetStepPos_Y = round(deltaY * stepsPerPixel_Y);

          // Apply deadband: only move if change is significant enough
          long deltaStepsX = abs(targetStepPos_X - currentStepPos_X);
          long deltaStepsY = abs(targetStepPos_Y - currentStepPos_Y);

          if (deltaStepsX > DEADBAND_STEPS_X && targetStepPos_X != currentStepPos_X) {
            enableMotorX();
            stepperX.moveTo(targetStepPos_X);
            lastMoveTimeX = millis();
            currentStepPos_X = targetStepPos_X;
          }

          if (deltaStepsY > DEADBAND_STEPS_Y && targetStepPos_Y != currentStepPos_Y) {
            enableMotorY();
            stepperY.moveTo(targetStepPos_Y);
            lastMoveTimeY = millis();
            currentStepPos_Y = targetStepPos_Y;
          }

          // Debug output (uncomment to see 2-axis angular compensation in action)
          /*
          Serial.print("Pan: ");
          Serial.print(targetAngleX, 1);
          Serial.print("° | Tilt: ");
          Serial.print(targetAngleY, 1);
          Serial.print("° | X corr: ");
          Serial.print((int)correctedX);
          Serial.print(" | Y corr: ");
          Serial.println((int)correctedY);
          */
          
          Serial.print("X=");
          Serial.print(targetStepPos_X);
          Serial.print(" Y=");
          Serial.println(targetStepPos_Y);
        }
      }
    }

    else {
      // Build up the message until we hit newline
      serialBuffer += incomingChar;
    }
  }
  
  // ========== WATCHDOG: Auto-disable laser if no message for 2 seconds ==========
  if (laser_state && (millis() - lastCoordTime > LASER_TIMEOUT)) {
    digitalWrite(Laser, LOW);
    laser_state = false;
    Serial.println("Laser OFF (timeout - no signal from Pi)");
  }

  // -------- HYBRID AUTO-DISABLE LOGIC --------

  // X motor: disable a bit after it finishes moving
  if (motorX_enabled) {
    if (stepperX.distanceToGo() == 0) {
      if (millis() - lastMoveTimeX > MOTOR_DISABLE_DELAY) {
        disableMotorX();
      }
    } else {
      lastMoveTimeX = millis();
    }
  }

  // Y motor: disable a bit after it finishes moving
  if (motorY_enabled) {
    if (stepperY.distanceToGo() == 0) {
      if (millis() - lastMoveTimeY > MOTOR_DISABLE_DELAY) {
        disableMotorY();
      }
    } else {
      lastMoveTimeY = millis();
    }
  }
}
