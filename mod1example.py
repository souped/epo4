import serial
import keyboard
import time

port = "COM4"

class KITT:
    def __init__(self, port, baudrate=115200) -> None:
        self.serial = serial.Serial(port, baudrate, rtscts=True)
        self.speed = 150
        self.angle = 150

    def send_command(self, command):
        self.serial.write(command.encode())

    def set_speed(self, speed):
        self.speed = speed
        self.send_command(f"M{speed}\n")
    
    def set_angle(self, angle):
        self.send_command(f"D{angle}\n")
    
    def stop(self):
        self.set_speed(150)
        self.set_angle(150)

    def ebrake(self):
        if self.speed > 150:
            self.set_speed(140)
            time.sleep(0.5)
            self.stop()
    
    def __del__(self):
        self.serial.close()
    
def wasd(kitt):
    def on_key_event(event):
        if event.event_type == keyboard.KEY_DOWN:
            match event.name:
                case "w":
                    kitt.set_speed(165)
                    print("forward")
                case "s":
                    kitt.set_speed(135)
                    print("backward")
                case "a":
                    kitt.set_angle(200)
                    print("turning left")
                case "d":
                    kitt.set_angle(100)
                    print("turning right")
                case "e":
                    kitt.send_command(b'A1\n')
                case "q":
                    kitt.send_command(b'A0\n')
        else:
            kitt.stop()
            print("stopping...")

if __name__ == "__main__":
    kitt = KITT(port)
    wasd(kitt)