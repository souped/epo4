import numpy as np
import math
from scipy.io import wavfile

from KITTMODEL import KITTMODEL
from KITT_communication import KITT
from keyboardfile import Keyboard
from Optimizing import localization
from statetracker import StateTracker
from microphone import Microphone

class RoutePlanner():
    def __init__(self, mod, state: StateTracker):
        self.mod = mod
        self.state = state

    def make_curve_route(self,carloc,cart_rad,dest):
        """
        This function generates the commands that get the car pointing to a destination.

        If the model cannot get the car pointing in the right direction, e.g. when the destination is too close
        to the car, drive it forwards, get its position and check again. Drive the car forwards until it can
        reach its destination.
        If the model goes out of bounds, drive the car backwards until it can reach its destination.'

        :param carloc: Location of the car as (x,y)
        :param cart_rad: Orientation of the car in rad
        :param dest: Destination as (x,y)
        :return: gen_cmd: The generated command as a string
        """
        # Generating commands
        end_pos, end_dir, gen_com, t = self.mod.generate_curve_command(carloc=carloc, cart_rad=cart_rad, dest=dest)
        while end_pos is None:
            print("No path possible from this location.")
            print("Driving forwards...")
            (x,y), actual_dir = self.state.after_straight_state(0)
            print("Recalculating...")
            end_pos, end_dir, gen_com, t = self.mod.generate_curve_command(carloc=(x,y), cart_rad=actual_dir, dest=dest)
        while end_pos == -1:
            print("Driving backwards...")
            (x,y), actual_dir = self.state.after_straight_state(1)
            print("Recalculating...")
            end_pos, end_dir, gen_com, t = self.mod.generate_curve_command(carloc=(x,y), cart_rad=actual_dir, dest=dest)
        gen_cmd = f"M158 D{gen_com} {t}"
        print("Generated curve command:", gen_cmd)
        return gen_cmd, end_pos, end_dir

    def make_straight_route(self,carloc,dest):
        """
        This function generates the commands that get the car to a destination over a straight line.
        :param carloc: Location of the car as (x,y)
        :param dest: Destination as (x,y)
        :return: Command that get the car to a destination over a straight line.
        """
        cmd = self.mod.generate_straight_command(carloc=carloc, dest=dest)
        print("Generated straight command:", cmd)
        return cmd


if __name__ == '__main__':
    import os
    if os.name == 'nt': # windows:
        sysport = 'COM2'
    elif os.name == 'posix':
        sysport = '/dev/cu.RNBT-3F3B'

    kitt = KITT(sysport)
    mod = KITTMODEL()
    loc = localization()
    mic = Microphone()
    Fref,ref_signal=wavfile.read("gold_codes/gold_code_ref13.wav")
    ref_signal=ref_signal[:,0]
    ref=ref_signal[8500:9000]
    state = StateTracker(kitt=kitt, mod=mod, loc = loc, mic=mic ,ref= ref)
    rp = RoutePlanner(mod, state)
    rp.make_curve_route(carloc=(1,0),cart_rad=0 * np.pi,dest=(0,3))
