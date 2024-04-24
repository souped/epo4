import serial
import time
import keyboard


# comport = "COM5"
# serial_port = serial.Serial(comport, 115200, rtscts=True)
#
# serial_port.write(b'D200\n')
# serial_port.write(b'S\n')
# status = serial_port.read_until(b'\x04')
# print(status)
# time.sleep(5)
# serial_port.close()

# beacon specs
carrier_freq = 10000    # Carrier frequency, min = 5 kHz & max = 30 kHz
bit_freq = 5000         # The bit frequency, min = 1 kHz & max = 5 kHz
repetition_cnt = 2500   # = bit_freq/repetition_freq, sets the time between transmissions, with repetition_freq 1-10 Hz.
codeword = 0xFEEDBACC   # Code word in hexadecimal
# speed specs
max_speed = 5


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
        self.prev_speed = 150  # Initialize previous speed to standstill

    def send_command(self, command):
        self.serial.write(command.encode())

    def set_speed(self, speed):
        self.prev_speed = speed  # Update previous speed
        self.send_command(f'M{speed}\n')

    def set_angle(self, angle):
        self.send_command(f'D{angle}\n')

    def stop(self):
        self.set_speed(150)
        self.set_angle(150)

    def emergency_brake(self):
        if self.prev_speed > 150:
            # If previous speed was greater than standstill, apply emergency brake
            self.set_speed(140)  # Set speed to move backwards
            time.sleep(0.5)  # Reverse for a short period
            self.stop()  # Stop the car

    def __del__(self):
        self.serial.close()     # safely closes the comport


def wasd(kitt):
    # checks for any keypress and what key is pressed.
    def on_key_event(event):
        # The statements below only run if the corresponding key is being pressed.
        if event.name == 'w' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_speed(150+max_speed)  # speed up the car forward
            print("fwd")
        elif event.name == 's' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_speed(150-max_speed)  # speed up the car reverse
            print("bw")
        elif event.name == 'a' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_angle(200)  # turn wheels fully left
            print("left")
        elif event.name == 'd' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_angle(100)  # turn wheels fully right
            print("right")
        elif event.name == 'e':
            kitt.send_command(b'A1\n')
            print("start beacon")
        elif event.name == 'q':
            kitt.send_command(b'A0\n')
            print("stop beacon")
        # Runs when the pressed key is released. Stops the car.
        elif event.name == 'w' or event.name == 's' or event.name == 'a' or event.name == 'd':
            kitt.stop()
            print("stop")

    keyboard.hook(on_key_event)     # check for any key status change
    kitt.emergency_brake()          # Check for emergency brake condition

if __name__ == "__main__":
    kitt = KITT("COM4")   # create KITT instance
    try:  # error handling
        wasd(kitt)  # keyboard function
    except Exception as e:
        print(e)
    keyboard.wait('ESC')  # Wait for 'q' key to exit
