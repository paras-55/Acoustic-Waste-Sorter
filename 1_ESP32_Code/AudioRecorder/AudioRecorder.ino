
#include <ESP32Servo.h>

Servo diverter;

void setup() {
  Serial.begin(115200);
  diverter.setPeriodHertz(50);
  diverter.attach(18, 500, 2500);
  diverter.write(90);
  delay(1000);

  Serial.println("Sorter ready");
  Serial.println("Send P for plastic, M for metal");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'P' || command == 'p') {
      diverter.write(45);
      Serial.println("Plastic position");
    } else if (command == 'M' || command == 'm') {
      diverter.write(135);
      Serial.println("Metal position");
    }
  }
}
