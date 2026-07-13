#include <WiFi.h>
#include <esp_now.h>

// 1 PC ESP32 MAC

uint8_t masterAddress[] = {0x8C, 0x4F, 0x00, 0xAA, 0xDB, 0x4C}; // PC MAC

typedef struct struct_message {
  char payload[100];
} struct_message;

struct_message incomingData;
struct_message outgoingData; // 2

// PC -> OT-2 Python
void OnDataRecv(const esp_now_recv_info_t* info, const uint8_t* data, int len) {
  memcpy(&incomingData, data, sizeof(incomingData));
  
  // OT-2 Python
  Serial.print("TO PC >>> ");
  Serial.println(incomingData.payload);
}

// 3
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // OT2
}

void setup() {
  Serial.begin(115200); 
  delay(1000);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed");
    return;
  }


  esp_now_register_recv_cb(OnDataRecv);

  // ESP32 v3.x
  esp_now_register_send_cb((esp_now_send_cb_t)OnDataSent);

  // 4 PC (Peer)
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, masterAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add PC master");
  } else {
    Serial.println("OT-2 slave initialized, [bidirectional] communication enabled!");
  }
}

void loop() {
  // OT-2 "DONE_OT2"
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();


    if (input.length() > 0) {
      input.toCharArray(outgoingData.payload, sizeof(outgoingData.payload));

      // PC
      esp_now_send(masterAddress, (uint8_t*)&outgoingData, sizeof(outgoingData));
    }
  }
}