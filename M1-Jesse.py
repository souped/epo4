import serial

comport = "COM5"
serial_port = serial.Serial(comport, 115200, rtscts=True)\

serial_port.write(b'D169\n')
serial_port.write(b'S\n')
status = serial_port.read_until(b'\x04')
print(status)
serial_port.close()
