from machine import Pin, PWM, I2C
import uasyncio as aio
import math, time, random

from vl53l0x import VL53L0X          
from run_time import mlp_act

DEADBAND = 0.30
PWM_DUTY_ON = 0.85
power_L = 1.0
power_R = 0.9
PULSE_MIN_US = 500
PULSE_MAX_US = 2500

# ---- Servo on GP9 (PWM) ----
servo = PWM(Pin(9))
servo.freq(50)  # 50 Hz

def _us_to_duty(us):
    return int(us * 65535 / 20000)

def write_servo_deg(deg):
    d = 0 if deg < 0 else (180 if deg > 180 else deg)
    us = PULSE_MIN_US + (PULSE_MAX_US - PULSE_MIN_US) * (d / 180.0)
    servo.duty_u16(_us_to_duty(us))
    
# ---- ToF (I2C0 on GP0/GP1) ----
shut = Pin(10, Pin.OUT, value=0)
time.sleep_ms(2)
shut.value(1)

time.sleep(0.02)

i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
print(i2c.scan())
tof = VL53L0X(i2c)
tof.start()

SEC_PER_DEG = 0.003
RAY_MAX_CM = 200.0

RAY_SEQUENCE = [-90, -45, 0, 45, 90, 45, 0, -45]
OBS_ANGLES = [-90, -45, 0, 45, 90]
ray_dists = {a: RAY_MAX_CM for a in RAY_SEQUENCE}
ray_idx = random.randrange(len(RAY_SEQUENCE))
last_servo_deg = 90

async def scan_dists():
    global ray_idx, last_servo_deg
    
    while True:
        deg = servo_deg_for_relative(RAY_SEQUENCE[ray_idx])
        
        wait_time = abs(last_servo_deg - deg) * SEC_PER_DEG;
        write_servo_deg(deg)
        
        await aio.sleep(wait_time)
        last_servo_deg = deg
        
        mm_sum = 0
        
        for _ in range(5):
            mm = tof.read()
            if mm is None or mm > 2000 or mm <= 0:
                mm = 2000
            mm_sum += mm
        
        sm = mm_sum / 5 / 10
        sm = min(200, max(0, sm))
        
        ray_dists[RAY_SEQUENCE[ray_idx]] = sm
        
        ray_idx = (ray_idx + 1) % len(RAY_SEQUENCE)
        
        if ray_idx % len(RAY_SEQUENCE) == 0:
            print(ray_dists)

# ---- Motor driver -----

PIN_AIN1 = 2
PIN_AIN2 = 3
PIN_PWMA = 4
PIN_BIN1 = 6
PIN_BIN2 = 5
PIN_PWMB = 7
PIN_STBY = 8
PWM_FREQ = 1000

cmd_l = 0
cmd_r = 0
cmd_hold = 0

def mkpwm(pin, freq=1000):
    p = PWM(Pin(pin)); p.freq(freq); p.duty_u16(0); return p

AIN1, AIN2 = Pin(PIN_AIN1, Pin.OUT, value=0), Pin(PIN_AIN2, Pin.OUT, value=0)
BIN1, BIN2 = Pin(PIN_BIN1, Pin.OUT, value=0), Pin(PIN_BIN2, Pin.OUT, value=0)
PWMA, PWMB = mkpwm(PIN_PWMA), mkpwm(PIN_PWMB)
STBY = Pin(PIN_STBY, Pin.OUT, value=1)

def _duty_from_frac(frac):
    if frac < 0: frac = 0
    if frac > 1: frac = 1
    return int(frac * 65535)

def set_tracks(qL, qR):
    dutyL = _duty_from_frac(PWM_DUTY_ON * power_L)
    dutyR = _duty_from_frac(PWM_DUTY_ON * power_R)
    # Left
    if qL > 0:   AIN1.value(1); AIN2.value(0); PWMA.duty_u16(dutyL)
    elif qL < 0: AIN1.value(0); AIN2.value(1); PWMA.duty_u16(dutyL)
    else:        AIN1.value(0); AIN2.value(0); PWMA.duty_u16(0)
    # Right
    if qR > 0:   BIN1.value(1); BIN2.value(0); PWMB.duty_u16(dutyR)
    elif qR < 0: BIN1.value(0); BIN2.value(1); PWMB.duty_u16(dutyR)
    else:        BIN1.value(0); BIN2.value(0); PWMB.duty_u16(0)
    
# ----- Other -----
    
def servo_deg_for_relative(rel_degrees):
    return 90 + rel_degrees
    
def quantize_ternary(x, deadband=DEADBAND):
    if x >  deadband: return  1
    if x < -deadband: return -1
    return 0

def build_observation():
    d = [min(1.0, max(0.0, ray_dists[a] / RAY_MAX_CM)) for a in OBS_ANGLES]
    return d + [cmd_l, cmd_r]

async def task_control():
    
    global cmd_l, cmd_r, cmd_hold
    period = 1.0 / 50.0
    
    while True:
        
        obs = build_observation()
        a_l, a_r = mlp_act(obs)
    
        ql = quantize_ternary(a_l)
        qr = quantize_ternary(a_r)
        
        if cmd_hold > 0:
            ql, qr = cmd_l, cmd_r
            cmd_hold -= 1
        else:
            if (ql != cmd_l) or (qr != cmd_r):
                cmd_l, cmd_r = ql, qr
                cmd_hold = 6
            
        set_tracks(cmd_l, cmd_r)
        await aio.sleep(period)
        
async def main():
    write_servo_deg(90)
    await aio.sleep(0.2)
    
    STBY.value(1)
    
    t1 = aio.create_task(scan_dists())
    t2 = aio.create_task(task_control())
    
    await aio.gather(t1, t2)
    
    
aio.run(main())