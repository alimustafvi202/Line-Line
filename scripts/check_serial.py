import serial
import time

try:
    s = serial.Serial()
    s.port = 'COM4'
    s.baudrate = 115200
    s.dtr = False
    s.rts = False
    s.timeout = 1.0
    s.open()
    time.sleep(3.0)
    raw = s.read(s.in_waiting or 2000)
    print("--- ESP32 COM4 OUTPUT ---")
    print(raw.decode('latin1', errors='replace'))
    print("-------------------------")
    s.close()
except Exception as e:
    print(f"Error opening COM4: {repr(e)}")
