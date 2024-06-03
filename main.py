import numpy as np
import matplotlib.pyplot as plt

from KITT_class_only import KITT
from Keyboard import wasd

sys_port = 'COM5'

if __name__ == '__main__':
    kitt = KITT(sys_port)   # Create KITT instance

    try:  # Error handling
        wasd(kitt, max_speed=10)  # Keyboard function
    except Exception as e:
        print(e)

    try:
        kitt.send_command()
        # collision_plotter(kitt)
    except Exception as e:
        print(e)

    # # When 'ESC' is pressed, exit. DOES NOT WORK!
    # while not keyboard.is_pressed('esc'):
    #     pass
