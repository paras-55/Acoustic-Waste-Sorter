
#include <ESP32Servo.h>

Servo diverter;

const int SERVO_PIN = 18;
const int PLASTIC_ANGLE = 45;
const int METAL_ANGLE = 135;

void setup() {
  Serial.begin(115200);

  diverter.setPeriodHertz(50);
  diverter.attach(SERVO_PIN, 500, 2500);
  diverter.write(90);

  delay(500);
  Serial.println("SORTER_READY");
}

void loop() {
  if (!Serial.available()) return;

  char command = Serial.read();

  if (command == 'P' || command == 'p') {
    diverter.write(PLASTIC_ANGLE);
    Serial.println("PLASTIC_OK");
  }
  else if (command == 'M' || command == 'm') {
    diverter.write(METAL_ANGLE);
    Serial.println("METAL_OK");
  }
  else if (command == 'C' || command == 'c') {
    diverter.write(90);
    Serial.println("CENTER_OK");
  }
}
