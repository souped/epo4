import numpy as np
import matplotlib.pyplot as plt
import time

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


speed_converter_lut = {
    5: self.m * 12.5,
    6: self.m * 15,
    7: self.m * 17.5,
    8: self.m * 20,
    9: self.m * 22.5,
    10: self.m * 25,
    11: self.m * 26.25,
    12: self.m * 27.5,
    13: self.m * 28.75,
    14: self.m * 30,
    15: self.m * 31.25,
}


def angle_converter_lut(angle):
    lut=[
        ((100,105),-25),
        ((105,115),-22),
        ((115,125),-19),
        ((125,135),-11),
        ((135,145),-9),
        ((145,155),0),
        ((155,165),5),
        ((165,175),7),
        ((175,185),10),
        ((185,195),17),
        ((195,200),20),
    ]

    for (start,end),value in lut:
        if start <= angle <= end:
            return value