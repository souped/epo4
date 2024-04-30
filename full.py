import serial
import time
import keyboard

class KITT:
    def __init__(self, port, baudrate=115200):
        self.serial = serial.Serial(port, baudrate, rtscts=True)
        self.set_speed(150)  # Default speed
        self.set_angle(150)  # Default angle

    def send_command(self, command):
        self.serial.write(command.encode())

    def set_speed(self, speed):
        self.send_command(f'M{speed}\n')

    def set_angle(self, angle):
        self.send_command(f'D{angle}\n')

    def stop(self):
        self.set_speed(150)
        self.set_angle(150)

    def close(self):
        self.stop()
        self.serial.close()

def wasd(kitt):
    def on_key_event(event):
        if event.name == 'w':
            kitt.set_speed(155)
        elif event.name == 's':
            kitt.set_speed(145)
        elif event.name == 'a':
            kitt.set_angle(200)
        elif event.name == 'd':
            kitt.set_angle(100)
        elif event.name == 'q':
            kitt.close()

    keyboard.on_press(on_key_event)
    keyboard.wait()

if __name__ == "__main__":
    kitt = KITT("COM5")
    wasd(kitt)
