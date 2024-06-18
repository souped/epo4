from kittfile import KITT
import time

class Keyboard:
    def car_model_input(kitt: KITT, input_cmd, from_curve = 0):
        """
        This function takes an input command and sends it to the car.
        :param kitt: The kitt class
        :param input_cmd: The input command in string format
        :return:
        """
        input_cmd = input_cmd.split(" ")
        for string in input_cmd:
            if "d" in string.lower():
                try:
                    direction = int(string[-3:])
                except ValueError:
                    print("------------------------ VALUEERROR ------------------------")
                    print(string)
                kitt.set_angle(direction)
            if "m" in string.lower():
                speed = int(string[-3:])
                kitt.set_speed(speed)
            else:
                pass
        ctime = float(input_cmd[-1])+1
        time.sleep(ctime)

        # To only forcefully stop the car when it is necessary.
        if from_curve == 0:
            kitt.emergency_brake(1)
        kitt.set_speed(150)

        # To completely get the steering to neutral.
        if direction < 150:
            kitt.set_angle(175)
            time.sleep(0.5)
            kitt.set_angle(150)
        elif direction > 150:
            kitt.set_angle(130)
            time.sleep(0.5)
            kitt.set_angle(150)
        else:
            kitt.set_angle(150)
        kitt.set_angle(150)