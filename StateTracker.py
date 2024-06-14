import numpy as np
from Keyboard import Keyboard
from Optimizing import localization
from microphone import Microphone


class StateTracker():
    def __init__(self, kitt, mod, loc: localization, mic: Microphone, ref):
        self.kitt = kitt
        self.mod = mod
        self.loc = loc
        self.mic = mic
        self.ref = ref

        self.current_pos = [0, 0]
        self.positions = []
        self.direction_rad = 0

    def determine_location(self):
        """
        Determines the current location of the car using the localization module.
        :return: x,y coordinates of the current location in m.
        """
        self.kitt.start_beacon()
        audio = self.mic.record_audio(seconds=2, devidx=self.mic.device_index)
        self.kitt.stop_beacon()
        print(type(audio))
        print(f"audio: {audio}")
        x,y = self.loc.localization(audiowav=audio,ref=self.ref)
        if 0 < x < 4.6 and 0 < y < 4.6:
            x,y = round(x/100, 5), round(y/100, 5)
            self.positions.append((x,y))
            return x,y
        else:
            x,y = self.determine_location()
            return x,y

    def after_curve_deviation(self,model_endpos,model_dir,dest,fwd_time=1.5,threshold=(0.3,0.5,1)):
        """
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
        x1, y1 = self.determine_location()
        print("Position 1: ", x1, y1)
        desired_pos_dev, _, _ = self.mod.desired_vector(model_endpos, (x1,y1))  # Calculate position deviation
        print('Car is currently', desired_pos_dev, "m away from predicted position")
        if desired_pos_dev > threshold[0]:
            Keyboard.car_model_input(kitt=self.kitt, input_cmd=f"M158 D150 {fwd_time}")  # Drive the car forwards for 1 second
            x2, y2 = self.determine_location()
            print("Position 2: ", x2, y2)

            # Calculations
            _, actual_dir, _ = self.mod.desired_vector((x1,y1), (x2,y2))  # Calculate the orientation deviation
            print('Car is currently', actual_dir, "rad from predicted orientation")
            length_to_dest, _, _ = self.mod.desired_vector((x2,y2), dest)  # Calculate distance to endpoint
            # Are all three conditions necessary? Should these be altered?
            if (desired_pos_dev < threshold[0] and np.abs(model_dir - actual_dir) < threshold[1] and
                    length_to_dest < threshold[2]):
                print("Car is on track!")
                self.direction_rad = actual_dir
                return 1, (x2,y2), actual_dir
            else:
                print("Car is off track!")
                self.direction_rad = actual_dir
                return 0, (x2,y2), actual_dir
        else:
            print("Car is close to destination")
            return 1, (x1,y1), model_dir

    def after_straight_state(self, direction, driving_time=2):
        """
        Determines the current direction of the car by moving either forwards or backwards.
        :param direction: 0 for forward, 1 for backward
        :return: (x,y) coordinates of the car in m, direction of the car in rad.
        """
        x1, y1 = self.determine_location()
        if direction == 0:
            Keyboard.car_model_input(kitt=self.kitt, input_cmd=f"D150 M158 {driving_time}")
            x2, y2 = self.determine_location()
            _, actual_dir, _ = self.mod.desired_vector((x1, y1), (x2, y2))  # Calculate the orientation deviation
        else:
            Keyboard.car_model_input(kitt=self.kitt, input_cmd=f"D150 M142 {driving_time}")
            x2, y2 = self.determine_location()
            _, actual_dir, _ = self.mod.desired_vector((x2, y2), (x1, y1))  # Calculate the orientation deviation
        self.direction_rad = actual_dir
        return (x2,y2), actual_dir

    # def weighted_location(self, model_endpos, model_dir, Tdoa_xy, Tdoa_dir):
    #     """
    #     Returns a weighted location of the car, depending on the generated model, current location and model time.
    #     :param model_endpos: End position of the model as (x,y)
    #     :param model_dir: End orientation of the model in rad
    #     :param Tdoa_xy: TDOA position as (x,y)
    #     :param Tdoa_dir: TDOA orientation in rad
    #     :return: Location of the car as (x,y)
    #     """
    #     if self.mod.modtime < 5:
    #         x = (model_endpos[0]*0.8 + Tdoa_xy[0]*0.2)
    #         y = (model_endpos[1] + Tdoa_xy[1])/2
    #         return x,y
    #     elif self.mod.modtime < 10:
    #         pass
    #     else:
    #         pass