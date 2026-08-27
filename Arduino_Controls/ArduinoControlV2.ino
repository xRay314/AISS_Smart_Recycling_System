#include <Servo.h>

Servo servo1; //paper
Servo servo2; //plastic
Servo servo3; //glass
Servo servo4; //metal

last_open=0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  servo1.attach(9); //paper
  servo1.write(90);

  servo2.attach(10); //plastic
  servo2.write(90);

  servo3.attach(11); // glass
  servo3.write(90);

  servo4.attach(6); //metal
  servo4.write(90);

  openServo(servo1);
  openServo(servo2);
  openServo(servo3);
  openServo(servo4);

}
 
void openServo(Servo &s) {
  s.write(0);   // open
}

void closeServos(Servo &s) {
  s.write(90);   // close
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    char data = Serial.read();

    if (data == '1') {
      openServo(servo1);
    }
    else if(data == '2') {
      openServo(servo2);
    }  
    else if (data == '3') {
      openServo(servo3);
    }  
    else if (data == '4') {
      openServo(servo4);
    }

    delay(2000);
    
  } else {
    closeServos(servo1);
    closeServos(servo2);
    closeServos(servo3);
    closeServos(servo4);
  }
   
}
  
