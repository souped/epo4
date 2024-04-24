import serial
import time

comport = "COM5"
serial_port = serial.Serial(comport, 115200, rtscts=True)\

serial_port.write(b'D200\n')
serial_port.write(b'S\n')
status = serial_port.read_until(b'\x04')
print(status)
sleep(5000)
serial_port.close()
