#define DIRPIN_1 14   // 1
#define STEPPIN_1 27  // 1
#define DIRPIN_2 18   // 2
#define STEPPIN_2 19  // 2
#define DIRPIN_3 33   // 3
#define STEPPIN_3 25  // 3
#include <ESP32Servo.h>  // ESP32

Servo servo_motor;
int pin = 16;       // -16


struct Motor {
  int speed;
  volatile bool running;
  bool direction;            // 1(true): /, 0(false): /
  volatile long pulseCount;  
  long targetPulseCount;
};

Motor motors[3];
volatile long absPosition[3] = {0, 0, 0};
bool systemRunning = false;


Motor sequence1[3] = { {500, false, false, 0, 2826}, {500, false, false, 0, 24600}, {500, false, false, 0, 500} };
Motor sequence2[3] = { {500, false, false, 0, 1000}, {500, false, false, 0, 1300}, {500, false, false, 0, 17100} };
Motor sequence3[3] = { {500, false, true, 0, 3826}, {500, false, true, 0, 25600}, {500, false, true, 0, 17600} };
Motor sequence4[3] = { {500, false, false, 0, 0}, {500, false, false, 0, 0}, {500, false, true, 0, 15800} };
Motor sequence5[3] = { {500, false, false, 0, 10100}, {500, false, false, 0, 500}, {500, false, false, 0, 0} };
Motor sequence6[3] = { {500, false, false, 0, 0}, {500, false, true, 0, 0}, {500, false, false, 0, 15826} };
Motor sequence7[3] = { {500, false, false, 0, 0}, {500, false, true, 0, 17800}, {500, false, true, 0, 3500} };
Motor sequence8[3] = { {500, false, false, 0, 14000}, {500, false, false, 0, 0}, {500, false, true, 0, 1500} };
Motor sequence9[3] = { {500, false, false, 0, 0}, {500, false, true, 0, 0}, {500, false, false, 0, 1500} };
Motor sequence10[3] = { {500, false, false, 0, 0}, {500, false, true, 0, 0}, {500, false, true, 0, 5500} };
Motor sequence11[3] = { {500, false, true, 0, 14640}, {500, false, true, 0, 0}, {500, false, true, 0, 0} };
Motor sequence12[3] = { {500, false, true, 0, 0}, {500, false, true, 0, 0}, {500, false, false, 0, 8000} };
Motor sequence13[3] = { {500, false, true, 0, 0}, {500, false, true, 0, 0}, {500, false, true, 0, 9000} };
Motor sequence14[3] = { {500, false, true, 0, 12500}, {500, false, false, 0, 2000}, {500, false, false, 0, 0} };
Motor sequence15[3] = { {500, false, true, 0, 0}, {500, false, true, 0, 0}, {500, false, true, 0, 5500} };
Motor sequence16[3] = { {500, false, false, 0, 13286}, {500, false, false, 0, 8600}, {500, false, false, 0, 7626} };

Motor* currentSequence = sequence1; 

void setup() {
  Serial.begin(115200);


  motors[0].speed = 400; 
  motors[1].speed = 400;
  motors[2].speed = 600; 

  pinMode(DIRPIN_1, OUTPUT); pinMode(STEPPIN_1, OUTPUT);
  pinMode(DIRPIN_2, OUTPUT); pinMode(STEPPIN_2, OUTPUT);
  pinMode(DIRPIN_3, OUTPUT); pinMode(STEPPIN_3, OUTPUT);

  servo_motor.attach(pin, 500, 2500);

  Serial.println("=== Minimal dial test mode started ===");
  Serial.println("Command dictionary:");
  Serial.println("  m 1 0 500 (fine tune: motor 1 direction 0 step 500)");
  Serial.println("  g 45      (gripper: move to 45 degrees)");
  Serial.println("  p         (read coordinates)");
  Serial.println("  z         (set zero point)");
  Serial.println("  h         (return to origin)");
  Serial.println("  s         (emergency stop!)");
  Serial.println("  load 1 / start (automation validation)");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // --- 1. () ---
    if (command == "s") { 
      stopAllMotors(); 
    } 
    // --- 2. ---
    else if (command == "p") {
      Serial.printf(">>> Coordinates [M1: %ld] [M2: %ld] [M3: %ld]\n", 
                    absPosition[0], absPosition[1], absPosition[2]);
    }
    else if (command == "z") {
      absPosition[0] = 0; absPosition[1] = 0; absPosition[2] = 0;
      Serial.println(">>> Zero point set (0, 0, 0)");
    }
    else if (command == "h") {
      Serial.println(">>> Returning to origin...");
      for (int i = 0; i < 3; i++) {
        motors[i].pulseCount = 0;
        if (absPosition[i] > 0) {
          motors[i].direction = false;
          motors[i].targetPulseCount = absPosition[i];
        } else if (absPosition[i] < 0) {
          motors[i].direction = true;
          motors[i].targetPulseCount = -absPosition[i];
        } else {
          motors[i].targetPulseCount = 0;
        }
        motors[i].running = true;
        digitalWrite(getDirPin(i), motors[i].direction ? HIGH : LOW);
      }
      systemRunning = true;
    }
    // --- 3. ---
    else if (command.startsWith("g ")) {
      int angle = command.substring(2).toInt();
      if (angle >= 0 && angle <= 180) {
        servo_motor.write(angle);
        Serial.printf(">>> Gripper: %d degrees\n", angle);
      } else {
        Serial.println("Error: 0~180");
      }
    }
    // --- 4. ---
    else if (command.startsWith("m ")) {
      int mIdx, dir; long steps;
      if (sscanf(command.c_str(), "m %d %d %ld", &mIdx, &dir, &steps) == 3) {
        mIdx -= 1;
        if (mIdx >= 0 && mIdx < 3) {
          motors[mIdx].direction = (dir == 1);
          motors[mIdx].pulseCount = 0;
          motors[mIdx].targetPulseCount = steps;
          motors[mIdx].running = true;
          systemRunning = true;
          digitalWrite(getDirPin(mIdx), motors[mIdx].direction ? HIGH : LOW);
          Serial.printf(">>> Running: M%d to direction %d for %ld steps\n", mIdx + 1, dir, steps);
        }
      } else {
        Serial.println("Format error! Example: m 1 1 500");
      }
    }
    // --- 5. ---
    else if (command.startsWith("load ")) { 
      loadSequence(command.substring(5).toInt()); 
    } 
    else if (command == "start") { 
      startMotors(); 
    } 
    else {
      Serial.println("Invalid command");
    }
  }

  // --- ---
  if (systemRunning) {
    bool anyRunning = false;
    for (int i = 0; i < 3; i++) {
      if (motors[i].running && motors[i].targetPulseCount > 0) {
        RunMotor(i);
        anyRunning = true;
      }
    }
    if (!anyRunning) systemRunning = false;
  }
}

/***** Stepper motor low-level driver and coordinate calculation *****/
void RunMotor(int motorIndex) {
  static unsigned long lastPulseTime[3] = {0, 0, 0};
  if (micros() - lastPulseTime[motorIndex] >= motors[motorIndex].speed) {
    digitalWrite(getStepPin(motorIndex), HIGH);
    delayMicroseconds(100);
    digitalWrite(getStepPin(motorIndex), LOW);
    lastPulseTime[motorIndex] = micros();

    motors[motorIndex].pulseCount++;

    if (motors[motorIndex].direction) {
      absPosition[motorIndex]++;
    } else {
      absPosition[motorIndex]--;
    }

    if (motors[motorIndex].pulseCount >= motors[motorIndex].targetPulseCount) {
      motors[motorIndex].running = false;
      Serial.printf("M%d in position, coordinate: %ld\n", motorIndex + 1, absPosition[motorIndex]);
    }
  }
}

/***** Helper functions *****/
void loadSequence(int seqNum) {
  Motor* targetSeq = nullptr;
  switch(seqNum) {
    case 1: targetSeq = sequence1; break; case 2: targetSeq = sequence2; break;
    case 3: targetSeq = sequence3; break; case 4: targetSeq = sequence4; break;
    case 5: targetSeq = sequence5; break; case 6: targetSeq = sequence6; break;
    case 7: targetSeq = sequence7; break; case 8: targetSeq = sequence8; break;
    case 9: targetSeq = sequence9; break; case 10: targetSeq = sequence10; break;
    case 11: targetSeq = sequence11; break; case 12: targetSeq = sequence12; break;
    case 13: targetSeq = sequence13; break; case 14: targetSeq = sequence14; break;
    case 15: targetSeq = sequence15; break; case 16: targetSeq = sequence16; break;
    default: Serial.println("Invalid sequence number"); return;
  }
  for (int i = 0; i < 3; i++) {
    motors[i] = targetSeq[i];
    Serial.printf("Loaded sequence%d - M%d: %ld steps\n", seqNum, i+1, motors[i].targetPulseCount);
  }
}

void startMotors() {
  systemRunning = true;
  for (int i = 0; i < 3; i++) {
    if (motors[i].targetPulseCount > 0) {
      motors[i].running = true;
      digitalWrite(getDirPin(i), motors[i].direction ? HIGH : LOW);
    } else {
      motors[i].running = false;
    }
  }
  Serial.println(">>> Array starting...");
}

void stopAllMotors() {
  for (int i = 0; i < 3; i++) motors[i].running = false;
  systemRunning = false;
  Serial.println("!!! Emergency stop !!!");
}

int getDirPin(int motorIndex) {
  int pins[] = {DIRPIN_1, DIRPIN_2, DIRPIN_3};
  return pins[motorIndex];
}

int getStepPin(int motorIndex) {
  int pins[] = {STEPPIN_1, STEPPIN_2, STEPPIN_3};
  return pins[motorIndex];
}