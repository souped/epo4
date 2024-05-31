import numpy as np
from KITTMODEL import KITTMODEL
from microphone import Microphone

def main():
    """setup audio connection"""
    dev_idx = Microphone.list_devices()

class Controller():
    def __init__(self) -> None:
        self.running = True

        # microphone
        self.recording_time = 4 # seconds
        self.mic = Microphone(channelnumbers = 8, Fs= 48000)
        self.stream = []

    def run_loop(self):
        while self.running is True:
            pass


if __name__ == "__main__":
    controller = Controller()
    controller.run_loop()

