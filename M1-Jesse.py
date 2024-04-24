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
    # checks for any keypress and what key is pressed.
    def on_key_event(event):
        # The statements below only run if the corresponding key is being pressed.
        if event.name == 'w' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_speed(155)  # speed up the car forward
            print("fwd")
        elif event.name == 's' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_speed(145)  # speed up the car reverse
            print("bw")
        elif event.name == 'a' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_angle(200)  # turn wheels fully left
            print("left")
        elif event.name == 'd' and event.event_type == keyboard.KEY_DOWN:
            kitt.set_angle(100)  # turn wheels fully right
            print("right")
        # Runs when the pressed key is released. Stops the car.
        elif event.name == 'w' or event.name == 's' or event.name == 'a' or event.name == 'd':
            kitt.stop()
            print("stop")
    # check for any key status change
    keyboard.hook(on_key_event)

if __name__=="__main__":
    kitt=KITT("COM4")   # create KITT instance
    wasd(kitt)  # keyboard function
    keyboard.wait('ESC')  # Wait for 'q' key to exit
