import numpy as np
import math

from KITTMODEL import KITTMODEL
from KITT_class_only import KITT
from Keyboard import Keyboard
from Optimizing import localization

sysport = 'COM4'

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


class RoutePlanner(KITTMODEL, KITT, Keyboard, localization):
    def __init__(self, max_angle_deg=25, tirewidth=5):
        KITTMODEL.__init__(self)
        KITT.__init__(self, sysport)

        self.mod = KITTMODEL
        self.kitt = KITT
        self.loc = localization

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

    def standstill_deviation(self, model_endpos, desired_dir, dest, fwd_time=1, threshold=(0.3, 0.5, 1)):
        """
        This function is not yet complete, as it misses TDOA implementation. The TDOA function should run for
        a couple of seconds to get an as accurate possible location measurements.

        This function checks if the car is deviating from the model path. It does this by driving the car forwards
        for a small time.
        This allows for calculating the cars actual orientation and checking the deviation in actual position.
        :param model_endpos: Final position of the model as (x,y)
        :param desired_dir: Final orientation of the model in rad
        :param dest: Destination as (x,y)
        :param fwd_time: Amount of time the car should go forwards to check for deviation.
        :param threshold: Thresholds for deviation. [0] = deviation in position, [1] = deviation in orientation, [2] = length to endpoint.
        :return: '1' if the car is on track, otherwise '0'
        """
        # Determine locations
        x1, y1 = self.tdoafunc()
        print("Position 1: ", x1, y1)
        desired_pos_dev, _, _ = self.desired_vector(model_endpos, (x1,y1))  # Calculate position deviation
        print('Car is currently', desired_pos_dev, "m away from predicted position")
        self.car_model_input(kitt=self.kitt, input_cmd=f"M157 D150 {fwd_time}")  # Drive the car forwards for 1 second
        x2, y2 = self.tdoafunc()
        print("Position 2: ", x2, y2)

        # Calculations
        _, actual_dir, _ = self.desired_vector((x1,y1), (x2,y2))  # Calculate the orientation deviation
        print('Car is currently', actual_dir, "rad from predicted orientation")
        length_to_dest, _, _ = self.desired_vector((x2,y2), dest)  # Calculate distance to endpoint
        # Are all three conditions necessary? Should these be altered?
        if (desired_pos_dev < threshold[0] and np.abs(desired_dir-actual_dir) < threshold[1] and
                length_to_dest < threshold[2]):
            print("Car is on track!")
            return 1
        else:
            print("Car is off track!")
            return 0

    def make_and_drive_route(self, carloc, cart_rad, dest):
        """
        This function gets and preforms the commands that get the car to a destination.

        If the model cannot get the car pointing in the right direction, e.g. when the destination is too close
        to the car, drive it forwards, get its position and check again. Drive the car forwards until it can
        reach its destination.

        It then sends the commands to the car and checks for deviation at the end of the curve.
        :param carloc: Location of the car as (x,y)
        :param cart_rad: Orientation of the car in rad
        :param dest: Destination as (x,y)
        :return:
        """
        # Generating commands
        end_pos, end_dir, gen_com, t = self.generate_curve_command(carloc=carloc, cart_rad=cart_rad, dest=dest)
        while end_pos is None:
            self.car_model_input(kitt=self.kitt, input_cmd="D150 M157 1")
            x, y = self.tdoafunc()
            end_pos, end_dir, gen_com, t = self.generate_curve_command(carloc=(x,y), cart_rad=cart_rad, dest=dest)
        gen_cmd = f"M157 D{gen_com} {t}"
        print("Generated curve command:", gen_cmd)

        # Send the command and check deviation
        self.car_model_input(kitt=self.kitt, input_cmd=gen_cmd)
        end_dir_rad = math.atan2(end_dir[1], end_dir[0])  # Convert unit vector to radians
        dev = self.standstill_deviation(model_endpos=end_pos,desired_dir=end_dir_rad, dest=dest)
        if dev == 1:
            print("Going straight.")
            print("Still need to make this functionality :)")
            # self.generate_straight_command()
        else:
            print("Recalculating commands.")
            self.make_and_drive_route(carloc, cart_rad, dest)


if __name__ == '__main__':
    rp = RoutePlanner()
    rp.make_and_drive_route(carloc=(1, 0), cart_rad=0 * np.pi, dest=(0, 1))
