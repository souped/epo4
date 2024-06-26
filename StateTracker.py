import numpy as np
from Keyboard import Keyboard
from Optimizing import localization


class StateTracker():
    def __init__(self, kitt, mod, loc: localization):
        self.kitt = kitt
        self.mod = mod
        self.loc = loc
        self.current_pos = [0, 0]
        self.positions = []
    """
    Should determine_location be placed in StateTracker or in the controller?
    """
    def determine_location(self):
        x,y = 0,0
        # x,y = self.loc.localize(audio,ref)
        current_pos = x,y
        self.positions.append(self.current_pos)
        return x,y

    def after_curve_deviation(self,model_endpos,model_dir,dest,fwd_time=1,threshold=(0.3,0.5,1)):
        """
        This function is not yet complete, as it misses TDOA implementation. The TDOA function should run for
        a couple of seconds to get an as accurate possible location measurements.

        This function checks if the car is deviating from the model path. It does this by driving the car forwards
        for a small time.
        This allows for calculating the cars actual orientation and checking the deviation in actual position.
        :param model_endpos: Final position of the model as (x,y)
        :param model_dir: Final orientation of the model in rad
        :param dest: Destination as (x,y)
        :param fwd_time: Amount of time the car should go forwards to check for deviation.
        :param threshold: Thresholds for deviation. [0] = deviation in position, [1] = deviation in orientation, [2] = length to endpoint.
        :return: '1' if the car is on track, otherwise '0'
        """
        # Determine locations
        self.kitt.start_beacon()
        x1, y1 = self.determine_location()
        self.kitt.stop_beacon()
        print("Position 1: ", x1, y1)
        desired_pos_dev, _, _ = self.mod.desired_vector(model_endpos, (x1,y1))  # Calculate position deviation
        print('Car is currently', desired_pos_dev, "m away from predicted position")
        Keyboard.car_model_input(kitt=self.kitt, input_cmd=f"M158 D150 {fwd_time}")  # Drive the car forwards for 1 second
        self.kitt.start_beacon()
        x2, y2 = self.determine_location()
        self.kitt.stop_beacon()
        print("Position 2: ", x2, y2)

        # Calculations
        _, actual_dir, _ = self.mod.desired_vector((x1,y1), (x2,y2))  # Calculate the orientation deviation
        print('Car is currently', actual_dir, "rad from predicted orientation")
        length_to_dest, _, _ = self.mod.desired_vector((x2,y2), dest)  # Calculate distance to endpoint
        # Are all three conditions necessary? Should these be altered?
        if (desired_pos_dev < threshold[0] and np.abs(model_dir - actual_dir) < threshold[1] and
                length_to_dest < threshold[2]):
            print("Car is on track!")
            return 1
        else:
            print("Car is off track!")
            return 0

    def after_straight_deviation(self,model_endpos,model_dir,dest):
        pass

    def weighted_location(self, model_endpos, model_dir, Tdoa_xy, Tdoa_dir):
        """
        Returns a weighted location of the car, depending on the generated model, current location and model time.
        :param model_endpos: End position of the model as (x,y)
        :param model_dir: End orientation of the model in rad
        :param Tdoa_xy: TDOA position as (x,y)
        :param Tdoa_dir: TDOA orientation in rad
        :return: Location of the car as (x,y)
        """
        if self.mod.modtime < 5:
            x = (model_endpos[0] + Tdoa_xy[0])/2
            y = (model_endpos[1] + Tdoa_xy[1])/2
            return x,y
        elif self.mod.modtime < 10:
            pass
        else:
            pass