import numpy as np
import math

from KITTMODEL import KITTMODEL
from KITT_communication import KITT
from Keyboard import Keyboard
from Optimizing import localization

sysport = 'COM5'

""""
This class does not yet follow the right construction as made on the whiteboard.
It currently acts as a main class, combining KITT, KITTMODEL and Keyboard.
The TDOA implementation is still missing.
Therefore it currently only supports generating and sending commands without deviation correction.

Inputs:
Starting x,y,theta coordinates
End x,y coordinates
Is this inputted as list? Or individual variables?

Output:
List of KITT commands
"""


class RoutePlanner():
    def __init__(self, kitt, mod, max_angle_deg=25, tirewidth=5):
        self.kitt = kitt
        self.mod = mod

        # Additional initialization
        self.max_angle_deg = max_angle_deg
        self.tirewidth = tirewidth
        self.min_rad = 33.5/np.sin(math.radians(max_angle_deg)) + tirewidth/2
        self.carloc = [0,0]
        self.dest = [0,0]

    def set_carloc(self, x, y):
        self.carloc = [x,y]

    def set_dest(self, x, y):
        self.dest = [x,y]

    def make_curve_route(self,carloc,cart_rad,dest):
        """
        This function generates the commands that get the car pointing to a destination.

        If the model cannot get the car pointing in the right direction, e.g. when the destination is too close
        to the car, drive it forwards, get its position and check again. Drive the car forwards until it can
        reach its destination.

        :param carloc: Location of the car as (x,y)
        :param cart_rad: Orientation of the car in rad
        :param dest: Destination as (x,y)
        :return: gen_cmd: The generated command as a string
        """
        # Generating commands
        end_pos, end_dir, gen_com, t = self.mod.generate_curve_command(carloc=carloc, cart_rad=cart_rad, dest=dest)
        while end_pos is None:
            print("Driving forwards...")
            Keyboard.car_model_input(kitt=self.kitt, input_cmd="D150 M158 3")
            x, y = self.tdoafunc()
            print("Recalculating...")
            end_pos, end_dir, gen_com, t = self.mod.generate_curve_command(carloc=(x,y), cart_rad=cart_rad, dest=dest)
        gen_cmd = f"M158 D{gen_com} {t*0.85}"
        print("Generated curve command:", gen_cmd)

        """
        # Send the command and check deviation
        Keyboard.car_model_input(kitt=self.kitt,input_cmd=gen_cmd)
        end_dir_rad=math.atan2(end_dir[1],end_dir[0])  # Convert unit vector to radians
        # dev = self.standstill_deviation(model_endpos=end_pos,desired_dir=end_dir_rad, dest=dest)
        # if dev == 1:
        #     print("Going straight.")
        #     print("Still need to make this functionality :)")
        #     # self.generate_straight_command()
        # else:
        #     print("Recalculating commands.")
        #     self.make_and_drive_route(carloc, cart_rad, dest)
        """
        return gen_cmd

    def make_straight_route(self,carloc,dest):

        cmd = self.mod.generate_straight_command(carloc=carloc, dest=dest)
        Keyboard.car_model_input(kitt=self.kitt,input_cmd=cmd)


if __name__ == '__main__':
    kitt = KITT(sysport)
    mod = KITTMODEL()
    rp = RoutePlanner(kitt,mod)
    rp.make_curve_route(carloc=(1,0),cart_rad=0 * np.pi,dest=(0,3))
