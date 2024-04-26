import serial
import time
import keyboard

# beacon specs
carrier_freq = 10000    # Carrier frequency, min = 5 kHz & max = 30 kHz
bit_freq = 5000         # The bit frequency, min = 1 kHz & max = 5 kHz
repetition_cnt = 2500   # = bit_freq/repetition_freq, sets the time between transmissions, with repetition_freq 1-10 Hz.
codeword = 0xFEEDBACC   # Code word in hexadecimal
# speed specs
max_speed = 10


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

    def send_command(self, command):
        if isinstance(command, str):
            command = command.encode()
        self.serial.write(command)

    def set_speed(self, speed):
        self.prev_speed = speed  # Update previous speed
        self.send_command(f'M{speed}\n')

    def set_angle(self, angle):
        self.send_command(f'D{angle}\n')

    def stop(self):
        self.set_speed(150)
        self.set_angle(150)

    def emergency_brake(self):  # STILL NEEDS TUNING!
        if self.prev_speed > 150:
            # If previous speed > standstill, apply emergency brake
            self.set_speed(140)     # Set speed to move backwards
            time.sleep(0.5)         # Reverse for a short period.
            self.stop()             # Stop the car
            print('Emergency Brake')
        elif self.prev_speed < 150:
            # If previous speed < standstill, apply emergency brake
            self.set_speed(160)  # Set speed to move backwards
            time.sleep(0.5)  # Reverse for a short period.
            self.stop()  # Stop the car
            print('Emergency Brake')

    def __del__(self):
        self.stop()             # In case the car was still moving, stop the car.
        self.serial.close()     # Safely closes the comport


def wasd(kitt):
    # Checks for any keypress and what key is pressed.
    def on_key_event(event):
        # The statements below only run if the corresponding key is being pressed.
        if event.name == 'w' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_speed(150+max_speed)  # speed up the car forward
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
        # Runs when the pressed key is released. Stops the car.
        elif event.name == 'w' or event.name == 's':
            kitt.emergency_brake()  # Check for emergency brake condition
            kitt.stop()


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
