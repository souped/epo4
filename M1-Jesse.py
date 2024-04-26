import serial
import time
import keyboard
import numpy as np

# beacon specs
carrier_freq = 10000    # Carrier frequency, min = 5 kHz & max = 30 kHz
bit_freq = 5000         # The bit frequency, min = 1 kHz & max = 5 kHz
repetition_cnt = 2500   # = bit_freq/repetition_freq, sets the time between transmissions, with repetition_freq 1-10 Hz.
codeword = 0xFEEDBACC   # Code word in hexadecimal
# speed specs
max_speed = 10

system_start = time.time()

last_event = {'type': None, 'name': None}

class KITT:
    def __init__(self, port, baudrate=115200):
        self.serial = serial.Serial(port, baudrate, rtscts=True)  # Open the comport

        # Initializing beacon specifications
        carrier_frequency = carrier_freq.to_bytes(2, byteorder='big')
        self.serial.write(b'F' + carrier_frequency + b'\n')
        bit_frequency = bit_freq.to_bytes(2, byteorder='big')
        self.serial.write(b'B' + bit_frequency + b'\n')
        repetition_count = repetition_cnt.to_bytes(2, byteorder='big')
        self.serial.write(b'R' + repetition_count + b'\n')
        code = codeword.to_bytes(4, byteorder='big')
        self.serial.write(b'C' + code + b'\n')

        # state variables such as speed, angle are defined here
        self.prev_speed = 150  # No speed

        self.distances = []

    def send_command(self, command):
        if isinstance(command, str):
            command = command.encode()
        self.serial.write(command)

    def read_command(self):
        start = time.time()
        self.send_command(b'Sd\n')
        message = self.serial.read_until(b'\x04')
        stop = time.time()
        message = message.decode()
        message = message[:-2]
        temp = message.replace("USL", "")
        temp=temp.replace("USR","")
        temp = temp.split("\n")
        temp = [int(i) for i in temp]
        temp.append(stop - system_start)
        temp.append(stop-start)
        self.distances.append(temp)
        print(self.distances[-1])
        self.serial.flush()

    def set_speed(self, speed):
        self.prev_speed = speed  # Update previous speed
        self.send_command(f'M{speed}\n')

    def set_angle(self, angle):
        self.send_command(f'D{angle}\n')

    def stop(self):
        self.set_speed(150)
        self.set_angle(150)

    def emergency_brake(self, from_speed):  # STILL NEEDS TUNING!
        if self.prev_speed > 153 and from_speed == 1:
            print('Emergency Brake')
            # If previous speed > standstill, apply emergency brake
            self.set_speed(140)     # Set speed to move backwards
            time.sleep(0.5)         # Reverse for a short period.
            self.stop()             # Stop the car
        elif self.prev_speed < 147 and from_speed == 1:
            print('Emergency Brake')
            # If previous speed < standstill, apply emergency brake
            self.set_speed(160)  # Set speed to move backwards
            time.sleep(0.5)  # Reverse for a short period.
            self.stop()  # Stop the car
        elif ((self.distances[-1][0] < 100 or self.distances[-1][1] < 100) and from_speed == 0
              and self.prev_speed != 150):
            print('Distance emergency Brake')
            self.set_speed(140)     # Set speed to move backwards
            time.sleep(0.5)         # Reverse for a short period.
            self.stop()             # Stop the car

    def __del__(self):
        self.stop()             # In case the car was still moving, stop the car.
        self.serial.close()     # Safely closes the comport

# check if keypress is the same, if true do nothing. otherwise
# recording function lags behind.
def wasd(kitt):
    # Checks for any keypress and what key is pressed.
    def on_key_event(event):
        if event.event_type == last_event['type']:
            kitt.read_command()
            kitt.emergency_brake(0)
        # The statements below only run if the corresponding key is being pressed.
        elif event.name == 'w' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_speed(150+max_speed)  # speed up the car forward
            print('forward')
        elif event.name == 's' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_speed(150-max_speed)  # speed up the car reverse
        elif event.name == 'a' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_angle(200)  # turn wheels fully left
        elif event.name == 'd' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_angle(100)  # turn wheels fully right
        elif event.name == 'e' and event.event_type == keyboard.KEY_DOWN:
            kitt.send_command(b'A1\n')
        elif event.name == 'q' and event.event_type == keyboard.KEY_DOWN:
            kitt.send_command(b'A0\n')
        elif event.name=='r' and event.event_type==keyboard.KEY_DOWN:
            kitt.read_command()
        # Runs when the pressed key is released. Stops the car.
        elif event.name == 'w' or event.name == 's':
            kitt.emergency_brake(1)  # Check for emergency brake condition
            kitt.stop()
        elif event.name == 'a' or event.name == 'd':
            kitt.set_angle(150)
        last_event['name'] = event.name
        last_event['type'] = event.event_type

    keyboard.hook(on_key_event)     # Check for any key status change

if __name__ == "__main__":
    kitt = KITT("COM5")   # Create KITT instance

    try:  # Error handling
        wasd(kitt)  # Keyboard function
    except Exception as e:
        print(e)
    # When 'ESC' is pressed, exit
    while not keyboard.is_pressed('esc'):
        pass
