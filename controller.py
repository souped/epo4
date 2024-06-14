import numpy as np
import os
from KITTMODEL import KITTMODEL
from microphone import Microphone
from Optimizing import localization
from scipy.io import wavfile
from Routeplanner import RoutePlanner
from KITT_communication import KITT
from keyboardfile import Keyboard
from statetracker import StateTracker
import time

REF_WAVFILE = "gold_codes/gold_code_ref13.wav"

if os.name == 'nt': # windows:
    sysport = 'COM2'
elif os.name == 'posix':
    sysport = '/dev/cu.RNBT-3F3B'
CHANNELS = 1
RATE = 48000

class Controller():
    def __init__(self) -> None:
        self.running = True

        self.md = KITTMODEL()
        self.kitt = KITT(sysport)
        self.localizer = localization()

        # microphone
        self.recording_time = 2 # seconds
        self.mic = Microphone(channelnumbers = CHANNELS, Fs = RATE)
        self.stream = []

        self.localiser = localization()

        # temporary reference signal
        Fref, ref_signal = wavfile.read(REF_WAVFILE)
        ref_signal =  ref_signal[:,0]
        self.ref = ref_signal[8500:9000]

        self.x, self.y = (0,0)
        self.state = StateTracker(self.kitt, self.md, self.localizer, self.mic, self.ref)
        self.rp = RoutePlanner(self.md, self.state)

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

        x,y = self.state.determine_location()
        print("Final location:", x,y)
        # track data?
        # do this inside the KITT Class or a separate other class i.e. DrivingHistory

        # temporary end
        self.running = False
        return x,y,dir

    def TDOA_tester(self):
        # input("Place car on the field, press Enter to continue...")
        i=0
        while i < 4:
            try:
                x,y=self.state.determine_location()
            except IndexError:
                x,y = -111,-111
            print("Current location:",x,y)
            time.sleep(2)
            i+=1

    def challenge_A(self):
        input_str = input("Enter destination coordinates (x,y):")
        x_str, y_str = input_str.split(',')
        x = float(x_str)
        y = float(y_str)
        self.run_loop(dest=(x,y))

    def challenge_B(self):
        input_str1 = input("Enter first destination coordinates (x,y):")
        x1_str, y1_str = input_str1.split(',')
        x1 = float(x1_str)
        y1 = float(y1_str)
        input_str2 = input("Enter first destination coordinates (x,y):")
        x2_str, y2_str = input_str2.split(',')
        x2 = float(x2_str)
        y2 = float(y2_str)
        carx, cary, cardir = self.run_loop(dest=(x1,y1))
        self.run_loop(dest=(x2,y2), carloc=(carx,cary), car_rad=cardir)

    def test_writer(self):
        self.kitt.start_beacon()
        audio = self.mic.record_audio(seconds=2, devidx=self.mic.device_index)
        self.kitt.stop_beacon()
        print(f"audio: {audio}")
        print(f"shape: {audio.shape}")
        self.mic.write_wavfile(audio, "failure.wav")


if __name__ == "__main__":
    controller = Controller()
    # controller.run_loop(dest=(4,2))
    # controller.test_writer()
    controller.TDOA_tester()