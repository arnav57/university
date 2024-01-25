void setup() {
  // put your setup code here, to run once:
  for (int i = 0; i < 10; i++) {
    pinMode(i, OUTPUT);
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  int del = 150;
  for (int i = 1; i < 10; i++) {
    digitalWrite(i, HIGH);
    delay(del);
    digitalWrite(i, LOW);
    delay(del);
  }
}
