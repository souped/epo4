import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer Qt5Agg
# enables interactive mode allowing us
# to dynamically plot new data on a single figure

"""
This module uses plt.ion() which leads to plt.show() not working as expected.
"""

plt.ion()


class DynamicPlotter:
    def __init__(self, xlabel="Time [s]", ylabel="Distance [m]"):
        self.figure, self.ax = plt.subplots(2, figsize=(15, 8))
        print("Initialising Dynamicplotter...")

        print("Enabling interactive mode...")
        # plt.ion()

        self.lines = []
        # get lines objects
        for x in self.ax:
            temp, = x.plot([], [])
            self.lines.append(temp)

        # configure plots
        for x in self.ax:
            # x.set_autoscaley_on(True)
            x.set_ylim([0, 300])
            x.set_autoscalex_on(True)

            x.set_title(f"Dynamic channel plot")
            x.set_xlabel(xlabel)
            x.set_ylabel(ylabel)

    def on_running(self, xdata, ydata, linesindex):
        # add new and old points
        self.lines[linesindex].set_xdata(xdata)
        self.lines[linesindex].set_ydata(ydata)
        for x in self.ax:
            x.relim()
            x.autoscale_view()
        self.figure.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def __del__(self):
        plt.close(self.figure)
        self.lines.clear()

