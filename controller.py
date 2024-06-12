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
from StateTracker import StateTracker
from GUI import GUI
import os

if os.name == 'nt': # windows:
    sysport = 'COM2'
elif os.name == 'posix':
    sysport = '/dev/cu.RNBT-3F3B'
CHANNELS = 8
RATE = 48000

class Controller():
    def __init__(self) -> None:
        self.running = True

        self.gui = GUI()
        self.md = KITTMODEL(self.gui)
        self.kitt = KITT(sysport)
        self.rp = RoutePlanner(self.kitt, self.md)
        self.localizer = localization()

        # microphone
        self.recording_time = 4 # seconds
        self.mic = Microphone(channelnumbers = CHANNELS, Fs= RATE)
        self.stream = []

        # Temporary reference signal
        Fref,ref_signal=wavfile.read("reference6.wav")
        ref_signal=ref_signal[:,1]
        self.ref=ref_signal[18800:19396]

        self.state = StateTracker(self.kitt, self.md, self.localizer, self.mic, self.ref, self.gui)

    def run_loop(self, dest, carloc=(0.2,0.3), car_rad=0.5*np.pi):
        # while self.running is True:
        # apply route planning algorithm?
        curve_cmd, model_endpos, model_dir = self.rp.make_curve_route(carloc, car_rad, dest)
        Keyboard.car_model_input(kitt=self.kitt, input_cmd=curve_cmd)

        state, (x,y), dir = self.state.after_curve_deviation(model_endpos=model_endpos, model_dir=model_dir, dest=dest)
        while state == 0:
            curve_cmd_corr,model_endpos,model_dir=self.rp.make_curve_route((x,y),dir,dest)
            Keyboard.car_model_input(kitt=self.kitt, input_cmd=curve_cmd_corr)
            state,(x,y),dir=self.state.after_curve_deviation(model_endpos=model_endpos,model_dir=model_dir,dest=dest)

        print("Generating straight commands...")
        straight_cmd = self.rp.make_straight_route(carloc, dest)
        Keyboard.car_model_input(kitt=self.kitt, input_cmd=straight_cmd)

        print("Final location:", self.state.determine_location())
        # track data?
        # do this inside the KITT Class or a separate other class i.e. DrivingHistory

        # send commands
        # self.kitt.send_cmd("cmd")

        # temporary end
        self.running = False

    def TDOA_tester(self):
        input("Place car on the field, press Enter to continue...")
        i = 0
        while i < 4:
            x, y = self.state.determine_location()
            print("Current location:", x, y)
            input("Move car to next location, press Enter to continue...")
            i += 1


if __name__ == "__main__":
    controller = Controller()
    controller.run_loop(carloc=(0.2,0.3), car_rad=0.5*np.pi, dest=(2,4))
    print(1)
