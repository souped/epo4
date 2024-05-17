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
        elif (kitt.prev_speed >= 150 or kitt.prev_speed < 150) and (
                kitt.distances[-1][0] < 250 or kitt.distances[-1][1] < 250):
            kitt.data.append(kitt.distances[-1])
            # plotter.on_running([j[2] for j in kitt.data], [j[0] for j in kitt.data], 0)
            # plotter.on_running([j[2] for j in kitt.data], [j[1] for j in kitt.data], 1)
            print("Getting data...")
        elif kitt.prev_speed==150 and (kitt.distances[-1][0] < 250 or kitt.distances[-1][1] < 250):
            # np.savetxt('distance_data.csv', kitt.data, delimiter=',')
            # kitt.data.clear()
            pass
        else:
            pass


def speed_converter(self, speed):
    if speed == 5:
        return self.m*12.5
    elif speed == 6:
        return self.m*15
    elif speed == 7:
        return self.m*17.5
    elif speed == 8:
        return self.m*20
    elif speed == 9:
        return self.m*22.5
    elif speed == 10:
        return self.m*25
    elif speed == 11:
        return self.m*26.25
    elif speed == 12:
        return self.m*27.50
    elif speed == 13:
        return self.m*28.75
    elif speed == 14:
        return self.m*30
    elif speed == 15:
        return self.m*31.25


def angle_converter(self, angle):
    if 100 <= angle <= 105:
        return -25
    elif 105 < angle <= 115:
        return -22
    elif 115 < angle <= 125:
        return -19
    elif 125 < angle <= 135:
        return -11
    elif 135 < angle <= 145:
        return -9
    elif 145 < angle <= 155:
        return 0
    elif 155 < angle <= 165:
        return 5
    elif 165 < angle <= 175:
        return 7
    elif 175 < angle <= 185:
        return 10
    elif 185 < angle <= 195:
        return 17
    else:
        return 20
