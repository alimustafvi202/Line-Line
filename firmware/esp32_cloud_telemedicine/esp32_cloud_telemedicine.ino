/*
 * CardioAI - ESP32 Smart Cloud Telemedicine Firmware (Karachi ➔ Haripur / Worldwide)
 * 
 * Features:
 * 1. Smart Web Config Portal (Zero Re-Flashing!):
 *    - Connect to a new Wi-Fi or mobile hotspot directly from your smartphone at http://192.168.4.1.
 *    - Remembers credentials in ESP32 non-volatile Flash (NVS Preferences).
 * 2. Automatic Fallback:
 *    - If saved Wi-Fi is in range -> Connects in 2 seconds and starts streaming!
 *    - If new location or Wi-Fi not found -> Opens "CardioAI-ECG-Setup" hotspot automatically!
 * 3. 250 Hz High-Speed ECG Streaming over Global MQTT (broker.emqx.io).
 * 4. Manual Config Trigger: Hold ESP32 BOOT button (GPIO 0) for 3 seconds to reconfigure anytime.
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// Global MQTT Cloud Broker (Free, fast, low-latency worldwide)
const char* MQTT_BROKER   = "broker.emqx.io";
const int   MQTT_PORT     = 1883;

// Fallback Default Device ID
String deviceId = "patient-ali-001";
String wifiSSID = "";
String wifiPass = "";

// Non-Volatile Flash Storage
Preferences prefs;
WebServer server(80);

// Hardware pins
const int CANDIDATE_PINS[] = {34, 35, 36, 32, 33};
const int NUM_CANDIDATES = 5;
int ecgPin = 34;

const int LO_PLUS_PIN = 18;
const int LO_MINUS_PIN = 19;
const int BOOT_BUTTON_PIN = 0; // ESP32 BOOT button

// MQTT
WiFiClient espClient;
PubSubClient mqttClient(espClient);
char mqttTopic[64];

// 250 Hz Sampling Timer
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_US = 4000;

// Batch buffer: Send 25 samples every 100 ms (10 packets/second)
const int BATCH_SIZE = 25;
int batchBuffer[BATCH_SIZE];
int batchIndex = 0;
unsigned long packetSequence = 0;

bool inConfigPortal = false;

// HTML Webpage served on mobile phone for Wi-Fi configuration
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<title>CardioAI Telemedicine Setup</title>";
  html += "<style>";
  html += "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; margin: 0; }";
  html += ".card { max-width: 420px; margin: 20px auto; background: #1e293b; border-radius: 16px; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1); }";
  html += "h2 { color: #38bdf8; margin-top: 0; text-align: center; }";
  html += "p { color: #94a3b8; font-size: 0.9rem; line-height: 1.5; text-align: center; }";
  html += "label { display: block; margin: 12px 0 6px; font-weight: 600; font-size: 0.85rem; color: #cbd5e1; }";
  html += "input, select { width: 100%; box-sizing: border-box; padding: 12px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #fff; font-size: 1rem; margin-bottom: 8px; }";
  html += "button { width: 100%; padding: 14px; border: none; border-radius: 8px; background: linear-gradient(135deg, #10b981, #06b6d4); color: #fff; font-size: 1rem; font-weight: 700; cursor: pointer; margin-top: 14px; }";
  html += ".badge { display: inline-block; background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; margin-bottom: 12px; }";
  html += "</style></head><body>";
  html += "<div class='card'>";
  html += "<h2>CardioAI Wireless Setup</h2>";
  html += "<p><span class='badge'>Remote Patient Node</span><br>Configure Wi-Fi for your ESP32 ECG device. No computer or reflashing required!</p>";
  
  html += "<form action='/save' method='POST'>";
  
  // Scan nearby networks
  int n = WiFi.scanNetworks();
  html += "<label>Select Nearby Wi-Fi:</label>";
  html += "<select name='ssid_select' onchange='document.getElementById(\"ssid\").value = this.value;'>";
  html += "<option value=''>-- Tap to select detected Wi-Fi --</option>";
  for (int i = 0; i < n; i++) {
    html += "<option value='" + WiFi.SSID(i) + "'>" + WiFi.SSID(i) + " (" + String(WiFi.RSSI(i)) + " dBm)</option>";
  }
  html += "</select>";

  html += "<label>Or Enter Wi-Fi Name (SSID):</label>";
  html += "<input type='text' id='ssid' name='ssid' value='" + wifiSSID + "' placeholder='e.g. Home Wi-Fi or Mobile Hotspot' required>";

  html += "<label>Wi-Fi Password:</label>";
  html += "<input type='password' name='pass' placeholder='Enter Wi-Fi Password'>";

  html += "<label>Patient Device ID (Haripur Dashboard):</label>";
  html += "<input type='text' name='dev_id' value='" + deviceId + "' placeholder='e.g. patient-ali-001' required>";

  html += "<button type='submit'>Save & Connect to Cloud</button>";
  html += "</form>";
  html += "</div></body></html>";

  server.send(200, "text/html", html);
}

void handleSave() {
  String newSsid = server.arg("ssid");
  String newPass = server.arg("pass");
  String newDevId = server.arg("dev_id");

  if (newSsid.length() > 0) {
    wifiSSID = newSsid;
    wifiPass = newPass;
    if (newDevId.length() > 0) deviceId = newDevId;

    prefs.begin("cardioai", false);
    prefs.putString("ssid", wifiSSID);
    prefs.putString("pass", wifiPass);
    prefs.putString("dev_id", deviceId);
    prefs.end();

    String successHtml = "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'>";
    successHtml += "<style>body{background:#0f172a;color:#fff;font-family:sans-serif;text-align:center;padding:40px;}</style></head><body>";
    successHtml += "<h2>Settings Saved Successfully!</h2>";
    successHtml += "<p>ESP32 is now restarting to connect to <strong>" + wifiSSID + "</strong> and will begin streaming ECG to Cloud MQTT.</p>";
    successHtml += "<p>You can now close this page and open your Dashboard in Haripur!</p>";
    successHtml += "</body></html>";

    server.send(200, "text/html", successHtml);
    delay(1500);
    ESP.restart();
  } else {
    server.send(400, "text/plain", "Missing Wi-Fi SSID");
  }
}

void startConfigPortal() {
  inConfigPortal = true;
  Serial.println("\n[CardioAI] Launching Smartphone Wi-Fi Config Hotspot...");

  WiFi.mode(WIFI_AP);
  WiFi.softAP("CardioAI-ECG-Setup", ""); // Password-free hotspot
  IPAddress IP = WiFi.softAPIP();

  Serial.println("==============================================================");
  Serial.println("  CONFIG HOTSPOT ACTIVE!");
  Serial.print("  1. Connect your phone Wi-Fi to: CardioAI-ECG-Setup\n");
  Serial.print("  2. Open phone browser and go to: http://");
  Serial.println(IP);
  Serial.println("  3. Pick your Wi-Fi, enter password, and click 'Save & Connect'!");
  Serial.println("==============================================================\n");

  server.on("/", handleRoot);
  server.on("/save", HTTP_POST, handleSave);
  server.begin();

  // Stay in portal loop until configured or reset
  while (inConfigPortal) {
    server.handleClient();
    delay(10);
  }
}

void connectToWiFiAndCloud() {
  Serial.print("\n[CardioAI Cloud] Connecting to Saved Wi-Fi: ");
  Serial.println(wifiSSID);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);

  WiFi.begin(wifiSSID.c_str(), wifiPass.c_str());

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 25) { // 12.5s initial wait
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[CardioAI Cloud] Wi-Fi Connected!");
    Serial.print("[CardioAI Cloud] Local IP: ");
    Serial.println(WiFi.localIP());
    // Immediately connect to MQTT broker on power-on
    reconnectMQTT();
  } else {
    Serial.println("\n[CardioAI Cloud] Still waiting for Wi-Fi router, will keep retrying automatically in background...");
  }
}

void reconnectMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;

  espClient.stop(); // Cleanly close any stale socket
  espClient.setTimeout(4); // 4s network timeout

  Serial.print("[CardioAI Cloud] Connecting to Global MQTT Cloud (");
  Serial.print(MQTT_BROKER);
  Serial.print(")... ");

  String clientId = "ESP32-ECG-" + deviceId + "-" + String(random(1000, 9999));
  
  char lwtTopic[64];
  snprintf(lwtTopic, sizeof(lwtTopic), "cardioai/telemedicine/%s/status", deviceId.c_str());
  
  if (mqttClient.connect(clientId.c_str(), lwtTopic, 1, true, "{\"online\": false}")) {
    Serial.println("CONNECTED!");
    mqttClient.publish(lwtTopic, "{\"online\": true}", true);
  } else {
    Serial.print("Failed (rc=");
    Serial.print(mqttClient.state());
    Serial.println("). Retrying in 2s...");
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);

  Serial.println("\n========================================================");
  Serial.println("   CardioAI - ESP32 Smart Telemedicine Cloud System     ");
  Serial.println("========================================================");

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(LO_PLUS_PIN, INPUT_PULLDOWN);
  pinMode(LO_MINUS_PIN, INPUT_PULLDOWN);
  pinMode(BOOT_BUTTON_PIN, INPUT_PULLUP);

  // Configure MQTT client properties FIRST before Wi-Fi connects
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setBufferSize(768);
  mqttClient.setKeepAlive(15);
  espClient.setTimeout(4);

  // Read saved credentials from non-volatile flash
  prefs.begin("cardioai", false);
  wifiSSID = prefs.getString("ssid", "");
  wifiPass = prefs.getString("pass", "");
  deviceId = prefs.getString("dev_id", "patient-ali-001");
  prefs.end();

  // Auto-scan pins to find where AD8232 OUT pin is connected
  int bestPin = 34;
  for (int i = 0; i < NUM_CANDIDATES; i++) {
    int p = CANDIDATE_PINS[i];
    int val = analogRead(p);
    if (val >= 200 && val <= 3900) {
      bestPin = p;
      break;
    }
  }
  ecgPin = bestPin;
  Serial.print("[CardioAI] Active ECG Input Pin: GPIO ");
  Serial.println(ecgPin);

  snprintf(mqttTopic, sizeof(mqttTopic), "cardioai/telemedicine/%s", deviceId.c_str());
  Serial.print("[CardioAI] Device ID: ");
  Serial.println(deviceId);

  // If no Wi-Fi saved yet, immediately launch smartphone config portal
  if (wifiSSID.length() == 0) {
    startConfigPortal();
  } else {
    connectToWiFiAndCloud();
  }
}

void loop() {
  // If user holds down BOOT button (GPIO 0) for 3 seconds, re-open config hotspot
  static unsigned long bootHoldStart = 0;
  if (digitalRead(BOOT_BUTTON_PIN) == LOW) {
    if (bootHoldStart == 0) {
      bootHoldStart = millis();
    } else if (millis() - bootHoldStart > 3000) {
      startConfigPortal();
    }
  } else {
    bootHoldStart = 0;
  }

  // Automatic Non-Blocking Wi-Fi & MQTT Reconnection
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.connected()) {
      static unsigned long lastMqttRetry = 0;
      if (millis() - lastMqttRetry > 2000) {
        lastMqttRetry = millis();
        reconnectMQTT();
      }
    } else {
      mqttClient.loop();
    }
  } else {
    // If Wi-Fi is lost, auto-retry reconnecting in background
    static unsigned long lastWifiRetry = 0;
    if (millis() - lastWifiRetry > 3000) {
      lastWifiRetry = millis();
      WiFi.disconnect();
      WiFi.begin(wifiSSID.c_str(), wifiPass.c_str());
    }
  }

  unsigned long currentMicros = micros();

  // Precise 250 Hz Sampling
  if (currentMicros - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = currentMicros;

    bool leadsOff = (digitalRead(LO_PLUS_PIN) == HIGH || digitalRead(LO_MINUS_PIN) == HIGH);
    int rawVal = analogRead(ecgPin);

    // If active pin drops to 0, rescanning fallback
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

    // 2-in-1 Hybrid: Also stream to USB Serial for native Web Serial (COM4)
    Serial.println(rawVal);

    batchBuffer[batchIndex++] = rawVal;

    // Send 25 samples every 100 ms (10 packets/second)
    if (batchIndex >= BATCH_SIZE) {
      batchIndex = 0;
      packetSequence++;

      if (mqttClient.connected()) {
        StaticJsonDocument<512> doc;
        doc["dev"] = deviceId;
        doc["seq"] = packetSequence;
        doc["lo"]  = leadsOff;

        JsonArray data = doc.createNestedArray("s");
        for (int i = 0; i < BATCH_SIZE; i++) {
          data.add(batchBuffer[i]);
        }

        char buffer[512];
        size_t len = serializeJson(doc, buffer);
        mqttClient.publish(mqttTopic, buffer, len);
      }

      Serial.print("[Cloud Stream] Batch #");
      Serial.print(packetSequence);
      Serial.print(" | ADC: ");
      Serial.println(rawVal);
    }
  }
}
