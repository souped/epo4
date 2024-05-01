import matplotlib.pyplot as plt

# enables interactive mode allowing us 
# to dynamically plot new data on a single figure
plt.ion()

class DynamicPlotter():
    def __init__(self, xlabel="Time [s]", ylabel="Sound magnitude?"):
        self.figure, self.ax = plt.subplots(5)
        
        self.lines = []
        # get lines objects
        for x in self.ax:
            temp, = x.plot([],[])
            self.lines.append(temp)


        # configure plots
        for x in self.ax:
            # x.set_autoscaley_on(True)
            x.set_ylim([0,10])
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
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
