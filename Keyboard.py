import keyboard
import numpy as np
import time

from KITT_communication import KITT


class Keyboard:
    def wasd(kitt: KITT, max_speed):
        # Checks for any keypress and what key is pressed.
        def on_key_event(event):
            if event.event_type == kitt.last_event['type'] and event.name == kitt.last_event['name']:
                kitt.emergency_brake(0)
            elif event.event_type == keyboard.KEY_DOWN:
                match event.name:
                    case "w":
                        kitt.set_speed(150 + max_speed)
                        print("Forwards")
                    case "s":
                        kitt.set_speed(150 - max_speed)
                        print("Backwards")
                    case "a":
                        kitt.set_angle(200)  # turn wheels fully left
                        print("Turning left")
                    case "d":
                        kitt.set_angle(100)  # turn wheels fully right
                        print("Turning right")
                    case "e":
                        kitt.start_beacon()
                    case "q":
                        kitt.stop_beacon()
                    case "r":
                        kitt.read_command()
                    case "p":
                        np.savetxt('distance_data.csv', kitt.data, delimiter=',')
                        print("Saved data")
                    case "o":
                        kitt.data.clear()
                        print("Cleared data")
                    case "i":
                        Keyboard.car_model_input(kitt)
                        return
            elif event.event_type == keyboard.KEY_UP:
                match event.name:
                    case "w" | "s":
                        kitt.emergency_brake(1)
                        kitt.set_speed(150)
                    case "a" | "d":
                        kitt.set_angle(153)
            kitt.last_event['type'] = event.event_type
            kitt.last_event['name'] = event.name

        keyboard.hook(on_key_event)  # Check for any key status change

    def car_model_input(kitt: KITT, input_cmd, from_curve = 0):
        """
        This function takes an input command and sends it to the car. If no input is given,
        it will ask the user to enter the command
        :param kitt: The kitt class
        :param input_cmd: The input command in string format
        :return:
        """
        cmd_string = input_cmd
        cmd_string = cmd_string.split(" ")
        print(cmd_string)
        for string in cmd_string:
            if "d" in string.lower():
                direction = int(string[-3:])
                # print("Set direction:", direction)
                kitt.set_angle(direction)
            if "m" in string.lower():
                speed = int(string[-3:])
                # print("Set speed:", speed)
                kitt.set_speed(speed)
            else:
                pass
        ctime = float(cmd_string[-1])
        # print("Time:", ctime)
        time.sleep(ctime)
        # print("Stop the car")
        if from_curve == 0:
            kitt.emergency_brake(1)

        kitt.set_speed(150)
        if direction < 150:
            kitt.set_angle(170)
            time.sleep(0.5)
            kitt.set_angle(150)
        elif direction > 150:
            kitt.set_angle(130)
            time.sleep(0.5)
            kitt.set_angle(150)
        else:
            kitt.set_angle(150)
        kitt.set_angle(150)
