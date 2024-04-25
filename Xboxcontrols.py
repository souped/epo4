# first do: !pip install pygame
# in your terminal otherwise this won't work!
import pygame
import serial
import time
import keyboard
import os

# Initialize Pygame and the joystick module
pygame.init()
pygame.joystick.init()

# Create a list to hold the connected joysticks
joysticks = []

# For all the connected joysticks
for i in range(0, pygame.joystick.get_count()):
    # Create a Joystick object in our list
    joysticks.append(pygame.joystick.Joystick(i))
    # Initialize the appended joystick
    joysticks[-1].init()
    # Print the name of the connected controller
    print("Detected joystick:", joysticks[-1].get_name())

# beacon specs
carrier_freq = 10000    # Carrier frequency, min = 5 kHz & max = 30 kHz
bit_freq = 5000         # The bit frequency, min = 1 kHz & max = 5 kHz
repetition_cnt = 2500   # = bit_freq/repetition_freq, sets the time between transmissions, with repetition_freq 1-10 Hz.
codeword = 0xFEEDBACC   # Code word in hexadecimal
# speed specs
max_speeding = 5
max_speed = -max_speeding+15


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
            self.set_speed(145)     # Set speed to move backwards
            time.sleep(0.5)         # Reverse for a short period.
            self.stop()             # Stop the car

    def __del__(self):
        self.stop()             # In case the car was still moving, stop the car.
        self.serial.close()     # Safely closes the comport


def xbox_controls(kitt):
    while True:
        # Check for events
        for event in pygame.event.get():
            # Check if the event is a joystick axis motion
            if event.type == pygame.JOYAXISMOTION:
                # Check which axis was moved
                if event.axis == 0:
                    # Left stick X-axis (steering)
                    # Convert axis value from range [-1, 1] to range [100, 200]
                    angle = int((event.value + 1) * 50 + 100)
                    if angle < 100:
                        angle = 100
                    kitt.set_angle(angle)
                    print("Angle set:", angle)
                elif event.axis == 4:
                    # Braking (LT)
                    # Convert axis value from range [-1, 1] to range [135, 150]
                    speed = int(-(event.value + 1) * 7.5 + 150 + max_speed)
                    if speed > 150:
                        speed = 150
                    kitt.set_speed(speed)
                    print("Speed set:", speed)
                elif event.axis == 5:
                    # Acceleration (RT)
                    # Convert axis value from range [-1, 1] to range [150, 165]
                    speed = int((event.value + 1) * 7.5 + 150 - max_speed)
                    if speed < 150:
                        speed = 150
                    kitt.set_speed(speed)
                    print("Speed set:", speed)
            # Check if the event is a button press
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:       # Turn beacon off when 'A' button is pressed
                    kitt.send_command(b'A0\n')
                    print("Beacon is off")
                elif event.button == 1:     # Turn beacon on when 'B' button is pressed
                    kitt.send_command(b'A1\n')
                    print("Beacon is on")

        kitt.emergency_brake()  # Check for emergency brake condition
        pygame.time.wait(10)    # Add a small delay to prevent high CPU usage
        keyboard.hook(exit_on_key('esc'))


def exit_on_key(keyname):   # since we are not actively listening to keyboard, we need to add a listen function.
    def callback(event):
        if event.name == keyname:
            os._exit(1)
    return callback


if __name__ == "__main__":
    kitt = KITT("COM4")   # Create KITT instance

    try:  # Error handling
        xbox_controls(kitt)  # Xbox controller function
    except Exception as e:
        print(e)

    # Quit Pygame
    pygame.quit()
