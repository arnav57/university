/*
SWITCHES PINOUT:
    SW0 - daytime? (Daylight Sensor)
    SW1 - people nearby? (Proximity Sensor)
    SW2 - within temp? (Temp Sensor)
    SW3 - within humidity? (Humid Sensor)
    SW4 - armed? (Toggle Switch)
    SW5 - door closed? (Hall sensor)

LED PINOUT:
    LED0 - Lighting System
    LED1 - Heater 
    LED2 - Humidifier
    LED3 - Alarm
*/


// required headers
# include <stdio.h>
# include <stdbool.h>

// define hardware pointer locations
# define SW_BASE        0xFF200040
# define LED_BASE       0xFF200000
# define TIMER_BASE     0xFFFEC600
# define SEVEN_SEG_BASE 0xFF200020   
# define GPIO_BASE      0xFF200060

// define hardware structures
typedef struct {
    volatile unsigned int * base;
} Switches;

typedef struct {
    volatile unsigned int * base;
} LED;

typedef struct {
    volatile unsigned int * base;
} Digit;

typedef struct {
    volatile unsigned int * load; // base addr
    volatile unsigned int * count;
    volatile unsigned int * control;
    volatile unsigned int * status;
} Timer;

typedef struct {
    volatile unsigned int * data; // base
    volatile unsigned int * control;
} GPIO;

// declare global variables w/ intial states
bool daytime = false, people_nearby = false, within_temp = false, within_humidity = false, armed = false, door_closed = false;

int main() {
    // define hardware pointers
    LED lite;
    LED * lights = &lite;
    lights -> base = (volatile unsigned int *) LED_BASE;

    Switches sw;
    Switches * switches = &sw;
    switches -> base = (volatile unsigned int *) SW_BASE;

    Digit d1;
    Digit * digit = &d1;
    digit -> base = (volatile unsigned int *) SEVEN_SEG_BASE;

    GPIO gp1;
    GPIO * gpio = &gp1;
    gpio -> data = (volatile unsigned int *) GPIO_BASE;
    gpio -> control = (volatile unsigned int *) (GPIO_BASE + 0x4);

    unsigned int testing_phase;

    while(1) {
        // check for current testing phase
        testing_phase = (*(switches -> base) & 0b1100000000) >> 8;
        // test phase 'A' is the FSM logic
        while (testing_phase == 0) {
            // check for testing phase update
            testing_phase = (*(switches -> base) & 0b1100000000) >> 8;

            // Phase 'A' of testing, print 'A' to the SevenSeg
            *(digit -> base) = 0b1110111;

            // start by reading switches (inputs), store them in global vars
            unsigned int sw_values = *(switches -> base);
            daytime = sw_values & 0b1;
            people_nearby = sw_values & 0b10;
            within_temp = sw_values & 0b100;
            within_humidity = sw_values & 0b1000;
            armed = sw_values & 0b10000;
            door_closed = sw_values & 0b100000;

            // lighting logic
            if (!daytime && people_nearby) {*(lights -> base) |= 0b1; }
            else {*(lights -> base) &= 0b1110; }

            // heater logic
            if (!within_temp) {*(lights -> base) |= 0b10;}
            else {*(lights -> base) &= 0b1101;}

            // humidifier logic
            if (!within_humidity) {*(lights -> base) |= 0b100;}
            else {*(lights -> base) &= 0b1011;}

            // alarm logic
            if (armed && !door_closed) {*(lights -> base) |= 0b1000;}
            else {*(lights -> base) &= 0b0111;}

            // delay for a few seconds
            delay(5);
	    }

        // test phase 'B' is the GPIO
        while(testing_phase == 1) {
            // check testing phase
            testing_phase = *(switches -> base) & 0b1100000000;
            // print 'B' to the seven seg
            *(digit -> base) = 0b1111100;
            // reset LEDs
            *(lights -> base) = 0;

            // config gpio as input/output
            // top half will be input, bottom half is output
            *(gpio -> control) = 0xFFFFFFFF;

            // write to output
            *(gpio -> data) = 1;
            *(lights->base) = 1;
            delay(5);
            *(gpio -> data) = 0;
            *(lights->base) = 0;
            delay(5);

        }

    }



}

void delay(unsigned int seconds) {
    Timer t1;
    Timer * timer = &t1;
    timer -> load = (volatile unsigned int *) TIMER_BASE;
    timer -> count = (volatile unsigned int *) (TIMER_BASE + 0x4);
    timer -> control = (volatile unsigned int *) (TIMER_BASE + 0x8);
    timer -> status = (volatile unsigned int *) (TIMER_BASE + 0xC); 

    // load timer
    *(timer -> load) = (unsigned int) seconds * 200000000;
    *(timer -> control) = 1; // only count down once.
	while( *(timer -> count) != 0 ) {
		; // when the count is not 0 do nothing
	}
}