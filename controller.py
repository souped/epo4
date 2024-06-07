# Libraries
import numpy as np
from scipy.io import wavfile

# Importing files
from KITTMODEL import KITTMODEL
from microphone import Microphone
from Optimizing import localization
from Routeplanner import RoutePlanner
from KITT_communication import KITT
from Keyboard import Keyboard

sysport = 'COM5'


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
        # ref_signal =  ref_signal[:,0]
        refsig = localization.detect_segments(ref_signal)
        self.ref = refsig[12][750:1500]

        self.x, self.y = (0,0)

        Fref,ref_signal=wavfile.read("Beacon/reference6.wav")
        ref_signal=ref_signal[:,1]
        self.ref=ref_signal[18800:19396]

    def run_loop(self, dest, carloc=(0,0), car_rad=0.5*np.pi):
        # while self.running is True:
        # apply route planning algorithm?
        self.rp.make_curve_route(carloc,car_rad,dest)

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
