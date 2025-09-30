# pico_car_tcp.py — Pico W TCP-controlled motors (MicroPython + uasyncio)
# Protocol (from phone/app):
#   Each line: "L R\n" where L,R ∈ [-100, 100]
#   Examples: "100 100\n" forward, "-100 -100\n" reverse, "0 0\n" stop
#   Extras: "STOP\n" forces stop, "PING\n" replies "PONG\n"

import network, uasyncio as asyncio
from machine import Pin, PWM
import time

# ========= MOTOR PINS (edit to your wiring) =========
PIN_AIN1 = 2
PIN_AIN2 = 3
PIN_PWMA = 4
PIN_BIN1 = 6
PIN_BIN2 = 5
PIN_PWMB = 7
PIN_STBY = 8
PWM_FREQ = 1000
# ====================================================

# ========= WIFI AP SETTINGS (edit if you like) ======
AP_SSID = "PicoCar"
AP_PASS = "pico-1234"     # >= 8 chars
AP_IP_HINT = "http://192.168.4.1/"  # just for your logs
# ====================================================

# ========= DRIVE / SAFETY TUNABLES ==================
WATCHDOG_MS = 2000        # stop if no command seen in this many ms
UPDATE_HZ   = 50          # motor update loop rate (Hz)
SCALE_L     = 0.88        # per-side scaling if one wheel is stronger
SCALE_R     = 1.0
INVERT_L    = False       # set True if your wiring makes it spin backwards
INVERT_R    = False
# ====================================================

# Onboard LED (optional heartbeat)
try:
    LED = Pin("LED", Pin.OUT)
except:
    LED = None

def mkpwm(pin, hz=PWM_FREQ):
    p = PWM(Pin(pin)); p.freq(hz); p.duty_u16(0); return p

# Pins/PWMs
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

def all_stop():
    motorA(0); motorB(0)

# ====== Shared state (updated by TCP task, consumed by driver) ======
left_cmd  = 0   # -100..100
right_cmd = 0
last_cmd_ms = time.ticks_ms()

def set_speeds(l, r):
    global left_cmd, right_cmd, last_cmd_ms
    # clamp
    t = r
    r = l
    l = t
    
    if l < -100: l = -100
    if l >  100: l =  100
    if r < -100: r = -100
    if r >  100: r =  100
    left_cmd, right_cmd = int(l), int(r)
    last_cmd_ms = time.ticks_ms()

async def drive_task():
    period = 1.0 / UPDATE_HZ
    STBY.value(1)
    print("Drive task: running at", UPDATE_HZ, "Hz; watchdog", WATCHDOG_MS, "ms")
    while True:
        # Watchdog
        if time.ticks_diff(time.ticks_ms(), last_cmd_ms) > WATCHDOG_MS:
            motorA(0); motorB(0)
        else:
            motorA(left_cmd)
            motorB(right_cmd)

        # heartbeat LED blink (slow)
        if LED:
            LED.toggle()
        await asyncio.sleep(period)

# ====== TCP CONTROL SERVER (port 9000) ======
# Accepts multiple clients; the last received line wins.
async def tcp_client(reader, writer):
    peer = None
    try:
        try:
            peer = writer.get_extra_info("peername")
        except:
            peer = None
        print("TCP client connected:", peer)
        await writer.awrite(b"HELLO PicoCar TCP. Send 'L R\\n' or 'STOP' or 'PING'.\n")
        while True:
            line = await reader.readline()
            if not line:
                break  # client closed
            try:
                s = line.decode().strip()
            except:
                continue
            if not s:
                continue

            if s.upper() == "STOP":
                set_speeds(0, 0)
                await writer.awrite(b"OK STOP\n")
                continue
            if s.upper() == "PING":
                await writer.awrite(b"PONG\n")
                continue

            # expect "L R"
            parts = s.split()
            if len(parts) != 2:
                await writer.awrite(b"ERR bad format\n")
                continue

            try:
                l = int(float(parts[0]))
                r = int(float(parts[1]))
            except:
                await writer.awrite(b"ERR parse\n")
                continue

            set_speeds(l, r)
            # Tiny ACK (optional). Comment out to reduce chatter.
            await writer.awrite(b"OK\n")
    except Exception as e:
        print("TCP client error:", e)
    finally:
        try:
            await writer.aclose()
        except:
            pass
        print("TCP client closed:", peer)

async def tcp_server_task():
    srv = await asyncio.start_server(tcp_client, "0.0.0.0", 9000, backlog=2)
    print("TCP control listening on 0.0.0.0:9000")
    await srv.wait_closed()

# ====== Wi-Fi AP bring-up ======
def start_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid=AP_SSID, password=AP_PASS)
    # Some firmwares need a moment to settle
    for _ in range(30):
        if ap.active():
            break
        time.sleep_ms(100)
    print("AP up:", ap.ifconfig(), "Open", AP_IP_HINT, "if you also serve HTTP.")
    return ap

# ====== OPTIONAL: tiny HTTP page just to show it's alive ======
# Comment this whole section if you don't want any HTTP.
async def http_client(reader, writer):
    try:
        # basic one-shot response with current L/R
        # drain request
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
        body = (
            "<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<h3>PicoCar TCP running</h3>"
            f"<p>Send TCP to <code>192.168.4.1:9000</code> lines like <code>100 100\\n</code>.</p>"
            f"<p>Current L,R: {left_cmd}, {right_cmd}</p>"
        )
        hdr = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        )
        await writer.awrite(hdr + body)
        await writer.drain()
    except:
        pass
    finally:
        try: await writer.aclose()
        except: pass

async def http_server_task():
    srv = await asyncio.start_server(http_client, "0.0.0.0", 80)
    print("HTTP status page on", AP_IP_HINT)
    await srv.wait_closed()

# ====== MAIN ======
async def main():
    start_ap()
    # Run: motors + TCP server (+ optional HTTP status)
    tasks = [drive_task(), tcp_server_task()]
    # comment next line if you don't want the status page
    tasks.append(http_server_task())
    print("Ready: control with a TCP client sending 'L R\\n' every ~250 ms.")
    await asyncio.gather(*tasks)

try:
    asyncio.run(main())
finally:
    all_stop()
    STBY.value(0)
    if LED: LED.value(0)
