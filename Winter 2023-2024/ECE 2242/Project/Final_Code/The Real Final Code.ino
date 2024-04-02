#include <DFRobot_OxygenSensor.h>
#define Oxygen_IICAddress ADDRESS_3
#define COLLECT_NUMBER  10 // collect number, the collection range is 1-100.
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_BME280.h>
#include <Omron_D6FPH.h> // change the header file to define I2C_ERROR_OK as 0, use MODEL_0505AD3
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Ticker.h>

// extra things on smonitor
#define PRINT_DEBUG false

// OLED panel macros
#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels

// BLE macros
#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331912a"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a0"
#define BLUETOOTH_NAME "VO2 Tracker Mask"


/*
By: Arnav, Josh, Graeme
For: ECE-2242B
Modified from Online Code

PINOUT
GP21 - SDA  (Serial Data)
GP22 - SCL (Serial Clock)
*/


/* SENSORS */
DFRobot_OxygenSensor oxygen;  // Declare a Oxy Sensor object
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1); // Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
Adafruit_BME280 bme;
bool status_bme;
Omron_D6FPH dps;

BLEServer *pServer;
BLEService *pService;
BLECharacteristic *pCharacteristic;
BLEAdvertising *pAdvertising;
Ticker updateTicker;

/* GLOBAL VARS */

// 'real-time' clock
#define RATE  5 // how many screen 'flickers'/updates per second
int counter = 0;
int seconds = 0;
int minutes = 0;

// sensor readings, note VolumeTotal holds VO2 Max
float O2, VO2, pressure;
float volumeTotal = 0.0;
float diffpressure = 0.0;

// VO2 Max calulation constants
float area_1 = 0.000531;   // = 26mm diameter
float area_2 = 0.000201; // uncomment for 16mm Venturi
float correctionSensor = 0.92; // for 16mm venturi measuresed with 3L calibration syringe
float TimerVolCalc;
float TimerStart = millis();
float TotalTime = 0;
int readVE = 0;
float TimerVE = 0.0;

// RGB LED Constants
// color channels
const int pwmR = 0;
const int pwmG = 1;
const int pwmB = 2;
// pins for each channel
const int Rpin = 12;
const int Gpin = 27;
const int Bpin = 14;
// RGB PWM constants
const int frequency = 1000;
const int resolution = 8;
// initial setup
void setup(void)
{
  // setup PWM for RGB LED
  ledcSetup(pwmR, frequency, resolution);
  ledcSetup(pwmG, frequency, resolution);
  ledcSetup(pwmB, frequency, resolution);

  ledcAttachPin(Rpin, pwmR);
  ledcAttachPin(Gpin, pwmG);
  ledcAttachPin(Bpin, pwmB);

  updateLED(2); // set LED to red to indicate setup
  
  // init serial
  Serial.begin(115200);

  // init OLED Panel
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 (OLED Screen) allocation failed"));
    while (1);
  }
  Serial.println("OLED Screen Connected");

  delay(2000);
  display.clearDisplay();
  display.setTextSize(0);
  display.setTextColor(WHITE); 

  // init Oxygen sensor
  while(!oxygen.begin(Oxygen_IICAddress)){
    Serial.println("Oxygen Sensor I2C Error");
    while (1);
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
  // setup BLE
  BLEDevice::init(BLUETOOTH_NAME);
  pServer = BLEDevice::createServer();
  pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                          CHARACTERISTIC_UUID,
                          BLECharacteristic::PROPERTY_READ |
                          BLECharacteristic::PROPERTY_NOTIFY
                      );
  pCharacteristic->setValue("Setting up..."); // Initial value
  pService->start();

  pAdvertising = pServer->getAdvertising();
  pAdvertising->addServiceUUID(pService->getUUID());
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06); 
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  updateTicker.attach(2, updateCharacteristic);
  Serial.print("Blutooth Server Enabled:");
  Serial.print(BLUETOOTH_NAME);
  Serial.print("\n");
}

// inf loop
void loop(void){
  TotalTime = millis() - TimerStart;

  O2 = oxygen.getOxygenData(COLLECT_NUMBER);
  updateBME();
  calculateVolume();
  updateOLED(O2, volumeTotal);

  Serial.print(volumeTotal/1000.0);
  Serial.print(" ,");
  Serial.print(O2);
  Serial.print("\n");
}

/*BLE FUNCTIONS */
// function called when updating BLE data push
void updateCharacteristic() {
  String newData = generateNewData(); // Function to generate new data
  std::string stdNewData = newData.c_str(); // Convert String to std::string
  pCharacteristic->setValue(stdNewData);
  pCharacteristic->notify(); // Notify connected devices about the change
}
// helper function to make BLE data push easier
String generateNewData() {
  // Example function to generate new data
  // Replace this with your own logic to generate the new data
  String newData = "VO2: " + String(volumeTotal) + " [mL]";
  return newData; // Random data for demonstration
}

/* UTIL FUNCTIONS - pretty self explanatory */

void calculateVolume(){
  float massFlow;
  float volFlow;
  float rho;
  float baro, temp;

  // calculate rho - density of dry air constant
  baro = bme.readPressure(); // in Pa
  temp = dps.getTemperature() + 273.15; // temp in K
  rho = (0.0289652 * baro) / (8.3144626 * temp); // nerd emoji

  TimerVolCalc = millis();

  // Read pressure from Omron D6F PH0025AD1 (or D6F PH0025AD2)
  float diffpressureraw = dps.getPressure()+50.0;

  float TimeElapsed = millis() - TimerVolCalc;
  diffpressure = diffpressure/2 + diffpressureraw/2 ;

  if (PRINT_DEBUG) {
    Serial.print("Differential pressure: ");
    Serial.print(diffpressure);
    Serial.print("\n");
    Serial.print("diff raw:");
    Serial.print(diffpressureraw);
    Serial.print("\n");
    Serial.print("pressure:");
    Serial.print(baro);
    Serial.print("\n");
    Serial.print("temp:");
    Serial.print(temp);
    Serial.print("\n");
  }

  
  //pply need break cases for if sensor is out of range
  if (diffpressure < 0) diffpressure = 0;

  if (diffpressure >= 0.2) { // ongoing integral of volumeTotal
    if (volumeTotal > 50) readVE = 1;
    updateLED(3); // set LED to green while recording
    float numerator = (abs(pressure) * 2 * rho) ;
    float denom = ((1 / (pow(area_2, 2))) - (1 / (pow(area_1, 2))));
    
    massFlow = 1000 * sqrt(numerator/denom); //Bernoulli equation
    
    volFlow = massFlow / rho; //volumetric flow of air
    //volFlow = volFlow * correctionSensor; // correction of sensor calculations
    volumeTotal = volFlow * TimeElapsed + volumeTotal;

    

  } else {
      updateLED(4); // set to Blue while looping but NOT reading;
      }
}


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
  //Calculate O2 Volume 

  display.setCursor(50,20);
  display.print(volumeValue / 1000.0);
  display.print(" L");
  display.display();
}

void updateBME() {
  pressure = bme.readPressure() / 100.0F ;
  if (PRINT_DEBUG) {
    Serial.print("Current Pressure [hPa]: ");
    Serial.print(pressure);
    Serial.print("\n");
  }
}

void updateDPS() { //Omron Differential
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

void updateLED(float Rdec, float Gdec, float Bdec) {

  float maxval = pow(2, resolution) - 1;
  int R = (int) Rdec/100.0 * maxval;
  int G = (int) Gdec/100.0 * maxval;  
  int B = (int) Bdec/100.0 * maxval;
  
  ledcWrite(pwmR, R);
  ledcWrite(pwmG, G);
  ledcWrite(pwmB, B);
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

void updateLED(int state_num) {
  /*
   *  0 - OFF
   *  1 - WHITE
   *  2 - RED
   *  3 - GREEN
   *  4 - BLUE
   */
  switch (state_num) {
    case 0:
      updateLED(0,0,0);
      break;
    case 1:
      updateLED(100,100,100);
      break;
    case 2:
      updateLED(100,0,0);
      break;
    case 3:
      updateLED(0,100,0);
      break;
    case 4:
      updateLED(0,0,100);
      break;
    default:
      updateLED(0,0,0);
  }
}