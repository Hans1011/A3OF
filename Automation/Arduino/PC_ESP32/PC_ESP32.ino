#include <WiFi.h>
#include <esp_now.h>

// ESP32 OT-2 MAC
uint8_t slaveAddress[] = {0x8C, 0x4F, 0x00, 0xA9, 0xD6, 0xC0};


typedef struct struct_message {
  char payload[100];
} struct_message;

struct_message incomingData;
struct_message outgoingData;

// ESP-NOW ()
void OnDataRecv(const esp_now_recv_info_t* info, const uint8_t* data, int len) {
  memcpy(&incomingData, data, sizeof(incomingData));
  
  // PC
  Serial.print("Wireless receipt from OT-2: ");
  Serial.println(incomingData.payload);
}

// ESP-NOW
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {

}

void setup() {
  Serial.begin(115200);  // PC
  delay(1000);
  WiFi.mode(WIFI_STA);

  Serial.println("ESP32 master started, [transparent wireless bridge mode] enabled");

  // ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed");
    return;
  }


  esp_now_register_recv_cb(OnDataRecv);
  
  // ESP32 Core v3.x API
  esp_now_register_send_cb((esp_now_send_cb_t)OnDataSent);

  // peer
  if (!esp_now_is_peer_exist(slaveAddress)) {
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, slaveAddress, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;

    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
      Serial.println("Failed to add slave");
    } else {
      Serial.println("Successfully bound OT-2 slave!");
    }
  }

  Serial.println("Listening ready: any string entered on PC serial will be directly forwarded to OT-2.");
}

void loop() {
  // PC -> ESP-NOW
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();


    if (input.length() > 0) {
      input.toCharArray(outgoingData.payload, sizeof(outgoingData.payload));

      // ESP32
      esp_err_t result = esp_now_send(slaveAddress, (uint8_t*)&outgoingData, sizeof(outgoingData));
      
      if (result == ESP_OK) {
        Serial.print("Wireless command transmitted: [");
        Serial.print(outgoingData.payload);
        Serial.println("]");
      } else {
        Serial.println("Send failed, please check if the ESP32 on OT-2 side is powered on");
      }
    }
  }
}