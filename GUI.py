import matplotlib.pyplot as plt
import numpy as np
import time

class GUI():
    def __init__(self):
        self.carloc = (0,0)
        self.cardir_rad = 0.5*np.pi
        self.arrow_endx = 0
        self.arrow_endy = 0
        self.dest = (0,0)
        self.path = [(0,0)]

        self.fig, self.ax = plt.subplots()


        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim([0, 4.6])
        self.ax.set_ylim([0, 4.6])

        # Create initial plots with empty data
        self.car_line, = self.ax.plot([], [], 'r-', label='Direction')
        self.car_point, = self.ax.plot([], [], 'ro', label='Car')
        self.path_line, = self.ax.plot([], [], label='Path')
        self.ax.legend()

    def update_carinfo(self, carloc = None, cardir = None):
        if carloc is not None:
            self.carloc = carloc
        if cardir is not None:
            self.cardir_rad = cardir

        arrow_length = 0.5
        self.arrow_endx = self.carloc[0] + arrow_length*np.cos(self.cardir_rad)
        self.arrow_endy = self.carloc[1] + arrow_length*np.sin(self.cardir_rad)
        self.show_plot()

    def update_path(self, path):
        self.path = path
        self.show_plot()

    def show_plot(self):
        # Update car line
        self.car_line.set_data([self.carloc[0], self.arrow_endx], [self.carloc[1], self.arrow_endy])

        # Update car point
        self.car_point.set_data([self.carloc[0]], [self.carloc[1]])

        # Update path
        self.path_line.set_data(*zip(*self.path))

        # Redraw the plot
        self.ax.set_xlim([0, 4.6])
        self.ax.set_ylim([0, 4.6])
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()


def generate_straight_path(start, end, num_points=100):
    """
    Generates a straight-line path from start to end with num_points points.
    Args:
    start (tuple): The starting (x, y) coordinates.
    end (tuple): The ending (x, y) coordinates.
    num_points (int): The number of points in the path.
    Returns:
    list: A list of (x, y) coordinates forming the path.
    """
    x_values = np.linspace(start[0], end[0], num_points)
    y_values = np.linspace(start[1], end[1], num_points)
    path = list(zip(x_values, y_values))
    print(path)
    return path

# Example usage
start_point = (0, 0)
end_point = (10, 10)
path = generate_straight_path(start_point, end_point)

if __name__ == '__main__':
    gui = GUI()
    # gui.show_plot()
    gui.update_carinfo(carloc = (1,2), cardir = 0.5*np.pi)
    plt.pause(0.1)
    time.sleep(2)
    gui.update_path(path)
    while True:
        plt.pause(0.1)