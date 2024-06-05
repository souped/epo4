import numpy as np
from KITTMODEL import KITTMODEL
from microphone import Microphone
from Optimizing import localization
from scipy.io import wavfile
from Routeplanner import RoutePlanner
from KITT_class_only import KITT
from Keyboard import Keyboard

sysport = 'COM5'


def main():
    """setup audio connection"""
    dev_idx = Microphone.list_devices()


class Controller():
    def __init__(self) -> None:
        self.running = True

        self.md = KITTMODEL()
        self.kitt = KITT(sysport)
        self.rp = RoutePlanner(self.kitt, self.md)

        # microphone
        self.recording_time = 2 # seconds
        self.mic = Microphone(channelnumbers = 8, Fs= 48000)
        self.stream = []

        self.localizer = localization()

        # temporary reference signal
        Fref, ref_signal = wavfile.read("reference.wav")
        ref_signal =  ref_signal[:,0]
        refsig = localization.detect_segments(ref_signal)
        self.ref = refsig[12][750:1500]

        self.x, self.y = (0,0)


    def run_loop(self):
        while self.running is True:
            # record audio
            pass

            # assuming beacon freq of 5 hz
            Fs, audio = wavfile.read("vanafxy-244-234.wav")
            print(audio.shape, self.ref.shape)

            # apply localisation algorithm
            # self.x, self.y = self.localiser.localization(audio, self.ref)

            # apply route planning algorithm?
            self.rp.make_and_drive_route()

            # track data?
            # do this inside the KITT Class or a separate other class i.e. DrivingHistory 

            # send commands 
            # self.kitt.send_cmd("cmd")
            Keyboard.car_model_input(self.kitt, inputstr="M160 D200 1")

            # temporary end
            self.running = False


if __name__ == "__main__":
    controller = Controller()
    controller.run_loop()
