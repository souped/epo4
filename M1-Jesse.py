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


class KITT:
    def __init__(self,port,baudrate=115200):
        self.serial=serial.Serial(port,baudrate,rtscts=True)
        # state variables such as speed, angle are defined here
        #self.send_command()

    def send_command(self,command):
        self.serial.write(command.encode())

    def set_speed(self,speed):
        self.send_command(f'M{speed}\n')

    def set_angle(self,angle):
        self.send_command(f'D{angle}\n')

    def stop(self):
        self.set_speed(150)
        self.set_angle(150)

    def __del__(self):
        self.serial.close()


def wasd(kitt):
    def on_key_event(event):
        if event.name=='w':
            kitt.set_speed(155)
        if event.name=='s':
            kitt.set_speed(145)
        if event.name=='a':
            kitt.set_angle(200)
        if event.name=='d':
            kitt.set_angle(100)
        if event.name=='q':
            kitt.__del__()
    keyboard.on_press(on_key_event)
    keyboard.wait()

if __name__=="__main__":
    # test code follows here
    kitt=KITT("COM5")
    wasd(kitt)
