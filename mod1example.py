import serial
import keyboard
import time
import sys

# beacon specs
carrier_freq = 10000    # Carrier frequency, min = 5 kHz & max = 30 kHz
bit_freq = 5000         # The bit frequency, min = 1 kHz & max = 5 kHz
repetition_cnt = 2500   # = bit_freq/repetition_freq, sets the time between transmissions, with repetition_freq 1-10 Hz.
codeword = 0xFEEDBACC   # Code word in hexadecimal

port = "/dev/cu.RNBT-3F3B"
keys = "wasdqe"

class KITT:
    def __init__(self, port, baudrate=115200) -> None:
        # self.serial = serial.Serial(port, baudrate, rtscts=True)
        self.speed = 150
        self.angle = 150
        self.boottime = time.time()

        # Beacon initialisation
        # carrier_frequency = carrier_freq.to_bytes(2, byteorder='big')
        # self.serial.write(b'F' + carrier_frequency + b'\n')
        # bit_frequency = bit_freq.to_bytes(2, byteorder='big')
        # self.serial.write(b'B' + bit_frequency + b'\n')
        # repetition_count = repetition_cnt.to_bytes(2, byteorder='big')
        # self.serial.write(b'R' + repetition_count + b'\n')
        # code = codeword.to_bytes(4, byteorder='big')
        # self.serial.write(b'C' + code + b'\n')

    def send_command(self, command: str):
        # self.serial.write(command.encode())
        print(command, "executed.")

    def read_command(self):
        t1 = time.time()
        self.send_command(b'Sd\n')
        # message = self.serial.read_until(b'\x04')
        t2 = time.time()
        # message = message.decode()
        testmessage = "USL140\nUSR120\n\x04"
        temp = testmessage.replace("USLR", "")
        temp = temp.split("\n")
        data = [int(i) for i in temp]
        data.append()
        self.distances.append(temp)
        print(f"responsetime: {t2-t1}")

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
                case "esc":
                    sys.exit(0)
        elif event.event_type == keyboard.KEY_UP and event.name in keys:
            kitt.stop()
            print("stopping...")

if __name__ == "__main__":
    kitt = KITT(port)   # Create KITT instance

    try:  # Error handling
        wasd(kitt)  # Keyboard function
        print(1)
    except Exception as e:
        print(e)

    # When 'ESC' is pressed, exit
    while not keyboard.is_pressed('esc'):
        pass