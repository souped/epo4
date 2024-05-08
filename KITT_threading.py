import serial
import time
import keyboard
import numpy as np
import asyncio
import matplotlib.pyplot as plt

from dynamicplotter import DynamicPlotter

sys_port = 'COM5'

# beacon specs
carrier_freq = 10000    # Carrier frequency, min = 5 kHz & max = 30 kHz
bit_freq = 5000         # The bit frequency, min = 1 kHz & max = 5 kHz
repetition_cnt = 2500   # = bit_freq/repetition_freq, sets the time between transmissions, with repetition_freq 1-10 Hz.
codeword = 0xFEEDBACC   # Code word in hexadecimal
# speed specs
max_speed = 10

system_start = time.time()


class KITT:
    def __init__(self, port, baudrate=115200):
        self.serial = serial.Serial(port, baudrate, rtscts=True)  # Open the comport

        # Initializing beacon specifications
        carrier_frequency = carrier_freq.to_bytes(2, byteorder='big')
        print(b'F' + carrier_frequency + b'\n')
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
        self.data = []

        self.last_event = {'type': None, 'name': None}

    def encode_command(self, command):
        if isinstance(command, str):
            command = command.encode()
        self.commands.append(command)

    async def send_command(self):
        while True:
            if not len(self.commands):
                start=time.time()
                self.serial.write(b'Sd\n')
                message=self.serial.read_until(b'\x04')
                stop=time.time()
                message=message.decode()
                message=message[:-2]
                temp=message.replace("USL","")
                temp=temp.replace("USR","")
                temp=temp.split("\n")
                temp=[int(i) for i in temp]
                temp.append(round((stop - system_start),3))
                temp.append(round((stop - start),3))
                self.distances.append(temp)
                print(self.distances[-1])
                self.serial.flush()
            else:
                self.serial.write(self.commands[0])
                self.commands.pop(0)
            await asyncio.sleep(0.01)

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
        if self.prev_speed > 153 and from_speed == 1:
            print('Emergency Brake')
            # If previous speed > standstill, apply emergency brake
            self.set_speed(140)     # Set speed to move backwards
            time.sleep(0.4)         # Reverse for a short period.
            self.set_speed(150)            # Stop the car
        elif self.prev_speed < 147 and from_speed == 1:
            print('Emergency Brake')
            # If previous speed < standstill, apply emergency brake
            self.set_speed(160)  # Set speed to move backwards
            time.sleep(0.4)  # Reverse for a short period.
            self.set_speed(150)  # Stop the car
        elif not self.distances:
            pass
        elif (((100 > self.distances[-1][0] > 0) or (100 > self.distances[-1][1] > 0)) and from_speed==0
              and self.prev_speed > 150):  # Brake because too close to an object
            print('Distance emergency Brake')
            self.set_speed(135)     # Set speed to move backwards
            time.sleep(0.3)         # Reverse for a short period.
            self.set_speed(150)            # Stop the car
        else:
            pass

    # def distance_emergency_brake(self):
    #     if (100 > self.distances[-1][0] > 0) or (100 > self.distances[-1][1] > 0) and self.prev_speed > 150:
    #         print('Distance emergency Brake')
    #         self.set_speed(135)  # Set speed to move backwards
    #         time.sleep(0.2)  # Reverse for a short period.
    #         self.set_speed(150)  # Stop the car
    #     else:
    #         pass

    def __del__(self):
        self.stop()             # In case the car was still moving, stop the car.
        self.serial.close()     # Safely closes the comport


def wasd(kitt):
    # Checks for any keypress and what key is pressed.
    def on_key_event(event):
        if event.event_type == kitt.last_event['type'] and event.name == kitt.last_event['name']:
            kitt.emergency_brake(0)
        elif event.event_type == keyboard.KEY_DOWN:
            match event.name:
                case "w":
                    kitt.set_speed(150+max_speed)
                    print("forward")
                case "s":
                    kitt.set_speed(150-max_speed)
                    print("backward")
                case "a":
                    kitt.set_angle(200)  # turn wheels fully left
                    print("turning left")
                case "d":
                    kitt.set_angle(100)  # turn wheels fully right
                    print("turning right")
                case "e":
                    kitt.start_beacon()
                case "q":
                    kitt.stop_beacon()
                case "r":
                    kitt.read_command()
                case "p":
                    np.savetxt('distance_data.csv',kitt.data,delimiter=',')
                    print("saved data")
                case "o":
                    kitt.data.clear()
        elif event.event_type == keyboard.KEY_UP:
            match event.name:
                case "w" | "s":
                    kitt.emergency_brake(1)
                    kitt.set_speed(150)
                case "a" | "d":
                    kitt.set_angle(151)
        kitt.last_event['type'] = event.event_type
        kitt.last_event['name'] = event.name

    keyboard.hook(on_key_event)     # Check for any key status change


async def collision_plotter(kitt):  # For the plotter to work with slowly driving towards the wall / stopping
    # the car in between, the emergency_brake from distance should be turned off.
    # plotter = DynamicPlotter()
    while True:
        if np.shape(kitt.data)[0] > 1000:
            print("Plotter has too much data!")
            print("Cleaning data...")
            kitt.data.clear()
        # REMOVE '=' IN FIRST COMPARISON!!!
        elif (kitt.prev_speed >= 150 or kitt.prev_speed < 150) and (kitt.distances[-1][0] < 250 or kitt.distances[-1][1] < 250):
            kitt.data.append(kitt.distances[-1])
            # plotter.on_running([j[2] for j in kitt.data], [j[0] for j in kitt.data], 0)
            # plotter.on_running([j[2] for j in kitt.data], [j[1] for j in kitt.data], 1)
            await asyncio.sleep(0.07)
        elif kitt.prev_speed == 150 and (kitt.distances[-1][0] < 250 or kitt.distances[-1][1] < 250):
            # np.savetxt('distance_data.csv', kitt.data, delimiter=',')
            # kitt.data.clear()
            await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.1)


async def kitt_main():
    kitt = KITT(sys_port)   # Create KITT instance

    try:  # Error handling
        wasd(kitt)  # Keyboard function
    except Exception as e:
        print(e)

    try:
        await asyncio.gather(kitt.send_command(), collision_plotter(kitt))
    except Exception as e:
        print(e)

    # When 'ESC' is pressed, exit. DOES NOT WORK!
    while not keyboard.is_pressed('esc'):
        pass

asyncio.run(kitt_main())
