from machine import Pin, PWM, I2C
import time
from vl53l0x import VL53L0X

# ---- Servo on GP9 (PWM) ----
servo = PWM(Pin(9))
servo.freq(50)  # 50 Hz

def deg_to_duty_us(deg):
    us = 500 + (2000 * (deg / 180))
    return int(us)

def write_servo_deg(deg):
    us = deg_to_duty_us(deg)
    duty = int(us * 65535 / 20000)
    servo.duty_u16(duty)
    
# ---- ToF (I2C0 on GP0/GP1) ----
shut = Pin(10, Pin.OUT, value=0)
time.sleep_ms(2)
shut.value(1)

i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
tof = VL53L0X(i2c)
tof.start()

SEC_PER_DEG = 0.002
SETTLE = 0.02

scan_points = [0, 45, 90, 135, 180]

# ---- Motor driver -----

PIN_AIN1 = 2
PIN_AIN2 = 3
PIN_PWMA = 4
PIN_BIN1 = 6
PIN_BIN2 = 5
PIN_PWMB = 7
PIN_STBY = 8
PWM_FREQ = 1000

AIN1, AIN2 = Pin(PIN_AIN1, Pin.OUT, value=0), Pin(PIN_AIN2, Pin.OUT, value=0)
BIN1, BIN2 = Pin(PIN_BIN1, Pin.OUT, value=0), Pin(PIN_BIN2, Pin.OUT, value=0)
PWMA, PWMB = mkpwm(PIN_PWMA), mkpwm(PIN_PWMB)
STBY = Pin(PIN_STBY, Pin.OUT, value=1)

def _duty_from_pct(pct):          # pct in [0..100] → 0..65535
    pct = 0 if pct < 0 else (100 if pct > 100 else int(pct))
    return (pct * 65535) // 100

def motorA(speed):                # speed in [-100..100]
    if INVERT_L:  speed = -speed
    if speed > 0:  AIN1.value(1); AIN2.value(0)
    elif speed < 0: AIN1.value(0); AIN2.value(1)
    else:           AIN1.value(0); AIN2.value(0)
    PWMA.duty_u16(_duty_from_pct(abs(int(speed * SCALE_L))))

def motorB(speed):
    if INVERT_R:  speed = -speed
    if speed > 0:  BIN1.value(1); BIN2.value(0)
    elif speed < 0: BIN1.value(0); BIN2.value(1)
    else:           BIN1.value(0); BIN2.value(0)
    PWMB.duty_u16(_duty_from_pct(abs(int(speed * SCALE_R))))