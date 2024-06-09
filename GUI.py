import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class GUI():
    def __init__(self):
        self.carloc = (0,0)
        self.cardir_rad = 0.5*np.pi
        self.dest = (0,0)
        self.path = (0,0)

        self.fig, self.ax = plt.subplots()
        self.fig_cardot = self.ax.plot(self.carloc[0], self.carloc[1], 'ro', label='Car')
        self.fig_cardir = self.ax.plot([], [], 'r-')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.legend()

    def update_carinfo(self, carloc, cardir):
        self.carloc = carloc
        self.fig_cardot.set_data(self.carloc[0], self.carloc[1])
        self.cardir = cardir

        arrow_length = 0.5
        arrow_endx = self.carloc[0] + arrow_length*np.cos(self.cardir)
        arrow_endy = self.carloc[1] + arrow_length*np.sin(self.cardir)
        self.fig_cardir.set_data([self.carloc[0], arrow_endx], [self.carloc[1], arrow_endy])

        self.fig.canvas.draw_idle()

    def update_path(self, path):
        self.path = path
        self.ax.plot(*zip(*self.path))

        self.fig.canvas.draw_idle()

    def show_plot(self):
        plt.show(block=False)

