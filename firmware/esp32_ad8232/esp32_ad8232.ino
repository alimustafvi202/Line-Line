/*
 * CardioAI - ESP32 + AD8232 ECG Smart Auto-Detect Firmware
 * 
 * Auto-detects whether your AD8232 OUT pin is plugged into GPIO 34, 35, 36 (VP), 32, or 33!
 * Prevents false flatlines and floating pin noise.
 */

const int CANDIDATE_PINS[] = {34, 35, 36, 32, 33};
const int NUM_CANDIDATES = 5;

// Optional hardware Leads-Off pins (connect AD8232 LO+ to GPIO 18, LO- to GPIO 19)
const int LO_PLUS_PIN = 18;
const int LO_MINUS_PIN = 19;

int ecgPin = 34; // Default to GPIO 34
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_US = 4000; // 250 Hz (4 ms)

void setup() {
  Serial.begin(115200);
  delay(400);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(LO_PLUS_PIN, INPUT_PULLDOWN);
  pinMode(LO_MINUS_PIN, INPUT_PULLDOWN);

  // Auto-scan pins to find where AD8232 output is connected (~1.65V -> ~1500-2500)
  int bestPin = 34;

  for (int i = 0; i < NUM_CANDIDATES; i++) {
    int p = CANDIDATE_PINS[i];
    int val = analogRead(p);
    
    // An active AD8232 sits near VCC/2 (approx 1200 - 2800)
    if (val >= 200 && val <= 3900) {
      bestPin = p;
      break;
    }
  }

  ecgPin = bestPin;
  Serial.print("\n[CardioAI] Active ECG Input Pin Auto-Detected: GPIO ");
  Serial.println(ecgPin);
  Serial.println("[CardioAI] 250 Hz Live Stream Ready\n");
}

void loop() {
  unsigned long currentMicros = micros();

  if (currentMicros - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = currentMicros;

    // Hardware Leads-Off detection (if LO+ or LO- are wired)
    if (digitalRead(LO_PLUS_PIN) == HIGH || digitalRead(LO_MINUS_PIN) == HIGH) {
      Serial.println("!LEADS_OFF");
      return;
    }

    int rawVal = analogRead(ecgPin);

    // If active pin drops to 0, check other pins once
    if (rawVal < 20) {
      for (int i = 0; i < NUM_CANDIDATES; i++) {
        int testVal = analogRead(CANDIDATE_PINS[i]);
        if (testVal > 250 && testVal < 3900) {
          ecgPin = CANDIDATE_PINS[i];
          rawVal = testVal;
          break;
        }
      }
    }

    Serial.println(rawVal);
  }
}
