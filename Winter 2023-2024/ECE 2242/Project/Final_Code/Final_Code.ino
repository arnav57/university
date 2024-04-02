#include <DFRobot_OxygenSensor.h>
#define Oxygen_IICAddress ADDRESS_3
#define COLLECT_NUMBER  10             // collect number, the collection range is 1-100.
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_BME280.h>
#include <Omron_D6FPH.h> // change the header file to define I2C_ERROR_OK as 0, use MODEL_0505AD3


#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels
#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define BLUETOOTH_NAME "VO2 Mask"


/*
By: Arnav, Josh, Graeme
For: ECE-2242B
Modified from Online Code

PINOUT
GP21 - 
GP22 - 
*/


/*DECLARE SENSORS*/
DFRobot_OxygenSensor oxygen;  // Declare a Oxy Sensor object
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1); // Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
Adafruit_BME280 bme;
bool status_bme;
Omron_D6FPH dps;



/*DECLARE GLOBAL VARS*/
#define RATE  5 // how many screen 'flickers'/updates per second
int counter = 0;
int seconds = 0;
int minutes = 0;

float O2, VO2, pressure, diffpressure;

// initial setup
void setup(void)
{
  // init serial
  Serial.begin(115200);

  // init OLED Panel
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 (OLED Screen) allocation failed"));
    for(;;);
  }
  Serial.println("OLED Screen Connected");

  delay(2000);
  display.clearDisplay();
  display.setTextSize(0);
  display.setTextColor(WHITE); 

  // init Oxygen sensor
  while(!oxygen.begin(Oxygen_IICAddress)){
    Serial.println("Oxygen Sensor I2C Error");
    delay(1000);
  }
  Serial.println("Oxygen Sensor Connected");

  // init barometer
  status_bme = bme.begin(0x76); 
  if (!status_bme) {
      Serial.println("Could not find a valid BME280 sensor, check wiring!");
      while (1);
  }
  Serial.println("BME Sensor Connected");
  
  // init diff pressure
  dps.begin(MODEL_0505AD3);
}

// inf loop
void loop(void)
{
  O2 = oxygen.getOxygenData(COLLECT_NUMBER);

  updateOLED(O2, 100.0);

  updateBME();
  
  updateDPS();
}

/*
BEGIN_FOLD
*/

// increment the current time
void getTime() {
  counter = counter + 1;

  if (counter%RATE == 0){
    counter = 0;
    seconds = (seconds + 1)%60;
    if (seconds >= 59) {
      minutes = minutes + 1;
    }
  }
}

void updateTime() {
  // dynamically update time m and s

  // get current time
  getTime();

  // write current time to display
  display.setCursor(75,0);
  display.println(minutes);
  display.setCursor(100,0);
  display.println(seconds);
}

void updateLabels() {
  // on initial setup write the labels
  display.setCursor(0,0);
  display.println("Time:");
  display.setCursor(0,10);
  display.println("O2:");
  display.setCursor(0,20);
  display.println("VO2:");
  display.display();
}

void updateO2(float oxygenValue) {
  // update the O2 Concentration value
  display.setCursor(50,10);
  display.print(oxygenValue);
  display.display();
}

void updateVO2(float volumeValue) {
  display.setCursor(50,20);
  display.print(volumeValue);
  display.print(" L");
  display.display();
}

void updateBME() {
  pressure = bme.readPressure() / 100.0F ;
  Serial.print("Current Pressure [hPa]: ");
  Serial.print(pressure);
  Serial.print("\n");
}

void updateDPS() {
  diffpressure = dps.getPressure();
  if(isnan(diffpressure)) {
    Serial.println("Could not read pressure data");
  }
  else {
    Serial.print("Current diff pressure: ");
    Serial.print(diffpressure);
    Serial.print("\n");
  }
}

void updateOLED(float o2, float vo2) {
  // clear the display
  display.clearDisplay();
  
  // start by writing the labels
  updateLabels();
  updateTime();
  updateO2(o2);
  updateVO2(vo2);

  // actually display the changes
  display.display();
}