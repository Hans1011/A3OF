/***** Pin definitions *****/
#define DIRPIN_3 18    // 3
#define STEPPIN_3 19   // 3

/***** Motor state variables *****/
int motorSpeed3 = 1200; // 500demo
bool motorRunning3 = false;
bool motorDirection3 = true;  // true: , false:

long pulseCount = 0;
long targetPulseCount = 0;


volatile bool motorFinished = false; 


void IRAM_ATTR countPulse() {
  if (motorRunning3) {
    pulseCount++;

    if (pulseCount >= targetPulseCount && targetPulseCount > 0) {  
      motorRunning3 = false;
      motorFinished = true;
    }
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(DIRPIN_3, OUTPUT);
  pinMode(STEPPIN_3, OUTPUT);

  attachInterrupt(digitalPinToInterrupt(STEPPIN_3), countPulse, RISING);
}

void loop() {

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // 1.
    if (command == "stop") {
      motorRunning3 = false;
      Serial.println(">>> Forced brake!");
    } 
    // 2.
    else if (command.startsWith("speed3:")) {
      motorSpeed3 = command.substring(7).toInt();
      Serial.print("Current speed (pulse interval) changed to: ");
      Serial.println(motorSpeed3);
    }
    // 3.  L  R
    else if (command.length() > 1) {
      char dirChar = command.charAt(0);
      long steps = command.substring(1).toInt();

      if ((dirChar == 'L' || dirChar == 'l' || dirChar == 'R' || dirChar == 'r') && steps > 0) {
        motorDirection3 = (dirChar == 'R' || dirChar == 'r');
        targetPulseCount = steps;
        pulseCount = 0;
        motorRunning3 = true;
        
        Serial.print("Received dial test command -> ");
        Serial.print(motorDirection3 ? "Right " : "Left ");
        Serial.print(steps);

        Serial.println(" steps (will automatically return completion message when done)"); 
      }
    }
  }


  if (motorRunning3) {
    RunMotor3();
  }


  if (motorFinished) {
    motorFinished = false;
    Serial.println(">>> Movement completed!"); 
  }
}

/*************************************************************/
/* Send pulse signal (keeping original absolute stable logic) */
/*************************************************************/
void RunMotor3() {
  digitalWrite(DIRPIN_3, motorDirection3 ? HIGH : LOW);
  static unsigned long lastPulseTime = 0;
  if (micros() - lastPulseTime >= motorSpeed3) {
    digitalWrite(STEPPIN_3, HIGH);
    delayMicroseconds(100);
    digitalWrite(STEPPIN_3, LOW);
    lastPulseTime = micros();
  }
}