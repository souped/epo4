import numpy as np
import matplotlib.pyplot as plt


def collision_plotter(kitt):  # For the plotter to work with slowly driving towards the wall / stopping
    # the car in between, the emergency_brake from distance should be turned off.
    # plotter = DynamicPlotter()
    while True:
        if np.shape(kitt.data)[0] > 1000:
            print("Plotter has too much data!")
            print("Cleaning data...")
            kitt.data.clear()
        # REMOVE '=' IN FIRST COMPARISON!!!
        elif (kitt.prev_speed >= 150 or kitt.prev_speed < 150) and (kitt.distances[-1][0] < 250 or kitt.distances[-1][1] < 250):
            kitt.data.append(kitt.distances[-1])
            # plotter.on_running([j[2] for j in kitt.data], [j[0] for j in kitt.data], 0)
            # plotter.on_running([j[2] for j in kitt.data], [j[1] for j in kitt.data], 1)
            print("Getting data...")
        elif kitt.prev_speed == 150 and (kitt.distances[-1][0] < 250 or kitt.distances[-1][1] < 250):
            # np.savetxt('distance_data.csv', kitt.data, delimiter=',')
            # kitt.data.clear()
            pass
        else:
            pass
