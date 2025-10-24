import serial
import time


# --- Initialize serial connection to Arduino ---
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)
time.sleep(3)  # wait for the serial connection to initialize
ser.reset_input_buffer()
print("Serial connection established")

try:
    while True:
        time.sleep(1)  # small delay to avoid overwhelming the serial port
        print("Send message to Arduino")
        ser.write("Hello from pi5!\n".encode('utf-8'))
except KeyboardInterrupt:
    print("Exiting serial read loop.")
    ser.close()