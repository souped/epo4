import serial
import time

# beacon specs
carrier_freq = 10000  # Carrier frequency, min = 5 kHz & max = 30 kHz, (5, 7500, 10
bit_freq = 5000  # The bit frequency, min = 1 kHz & max = 5 kHz, (1 , 2.5 5
# rep 250 for 1k, 625 for 2.5k, 1250 for 5k
repetition_cnt = 1250  # = bit_freq/repetition_freq, sets the time between transmissions, with repetition_freq 1-10 Hz.
codeword = 0xEA28B7C5  # Code word in hexadecimal

system_start = time.time()  # initialize time


class KITT:
    def __init__(self, port, baudrate=115200):
        self.serial = serial.Serial(port, baudrate, rtscts=True)  # Open the comport

        # Initializing beacon specifications
        carrier_frequency = carrier_freq.to_bytes(2, byteorder='big')
        # print(b'F' + carrier_frequency + b'\n')
        self.serial.write(b'F' + carrier_frequency + b'\n')
        bit_frequency = bit_freq.to_bytes(2, byteorder='big')
        self.serial.write(b'B' + bit_frequency + b'\n')
        repetition_count = repetition_cnt.to_bytes(2, byteorder='big')
        self.serial.write(b'R' + repetition_count + b'\n')
        code = codeword.to_bytes(4, byteorder='big')
        self.serial.write(b'C' + code + b'\n')

        # state variables such as speed, angle are defined here
        self.prev_speed = 150  # No speed

        self.distances = []  # left, right, system time, data age since command is send
        self.commands = []  # List where commands are stored that need to be sent to KITT

        self.last_event = {'type': None, 'name': None}
        print("Connected to car!")

    def encode_command(self, command):
        if isinstance(command, str):
            command = command.encode()
        self.commands.append(command)
        # Comment line below if you want keyboard input
        self.send_command()

    def loop_command(self):
        while True:
            if not len(self.commands):
                time.sleep(0.1)
            else:
                self.serial.write(self.commands[0])
                # print("Command sent")
                self.commands.pop(0)
                time.sleep(0.05)

    def send_command(self):
        self.serial.write(self.commands[0])
        print(f"Command sent: {self.commands[0]}")
        self.commands.pop(0)

    def set_speed(self, speed):
        self.prev_speed = speed  # Update previous speed
        self.encode_command(f'M{speed}\n')

    def set_angle(self, angle):
        self.encode_command(f'D{angle}\n')

    def stop(self):
        self.set_speed(150)
        self.set_angle(151)

    def start_beacon(self):
        self.encode_command(b'A1\n')

    def stop_beacon(self):
        self.encode_command(b'A0\n')

    def emergency_brake(self, from_speed):  # STILL NEEDS TUNING!
        if self.prev_speed > 153 and from_speed==1:
            print('Speed emergency Brake')
            # If previous speed > standstill, apply emergency brake
            self.set_speed(140)  # Set speed to move backwards
            time.sleep(0.4)  # Reverse for a short period.
            self.set_speed(150)  # Stop the car
        elif self.prev_speed < 147 and from_speed==1:
            print('Speed emergency Brake')
            # If previous speed < standstill, apply emergency brake
            self.set_speed(160)  # Set speed to move backwards
            time.sleep(0.4)  # Reverse for a short period.
            self.set_speed(150)  # Stop the car
        elif not self.distances:
            pass
        elif (((50 > self.distances[-1][0] > 0) or (50 > self.distances[-1][1] > 0)) and from_speed==0
              and self.prev_speed > 150):  # Brake because too close to an object
            print('Distance emergency Brake')
            self.set_speed(135)  # Set speed to move backwards
            time.sleep(0.3)  # Reverse for a short period.
            self.set_speed(150)  # Stop the car
        else:
            pass

    def __del__(self):
        self.serial.close()  # Safely closes the comport
