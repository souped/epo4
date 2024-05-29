import matplotlib.pyplot as plt
import numpy as np
import regex as re


class KITTMODEL():
    def __init__(self) -> None:
        self.m = 5.6  # mass, [kg]
        self.b = 5  # viscous friction, [N m^-1 s]
        self.c = 0.1  # air drag, [N m^-2 s^2]
        self.Famax = 400  # Max acc force, [N]
        self.Fbmax = 500  # max brake force, [N]
        self.L = 33.5  # Length of wheelbase


        self.figure = plt.figure()
        self.ax = self.figure.subplots()
        self.figure2 = plt.figure()
        self.ax2 = self.figure2.subplots()
        self.ax.grid()
        plt.show(block=False)

        # data
        self.positions = [(0,0)]
        self.velocities = [0]
        self.times = [0]

        # self.xy = [(0,0), (1,1), (2,2), (4,4), (6,7), (9,0)]

        self.lines, = self.ax.plot(*zip(*self.positions))
        self.vellines, = self.ax2.plot(self.times, self.velocities)
        # self.lines, = self.ax.plot(self.velocities)

        # state variables
        self.v = 0
        self.a = 0
        self.phi = 0
        self.direction = [1, 0]
        self.pos = (0, 0)
        self.t = 0
        self.dt = 0.01
        self.f = None

        

    def update_line(self):
        self.lines.set_data(*zip(*self.positions))
        self.vellines.set_ydata(self.velocities)
        self.vellines.set_xdata(self.times)
        # self.lines.set_data(self.velocities)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_aspect('equal')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.figure2.canvas.draw()
        self.figure2.canvas.flush_events()
        
    def sim(self, inputs):
        i = 0
        for x in inputs:
            # process input
            time = self.proc_cmd(x)

            # simulate input
            for t in np.arange(0, time, self.dt):
                print("F:", self.f)
                self.v = self.velocity(self.dt, self.f)
                print("V: ", self.v)
                self.velocities.append(self.v)
                self.direction = self.det_rotation()
                self.pos = self.det_xy(self.dt)
                self.positions.append(self.pos)
                self.t += self.dt
                self.times.append(self.t)

            # update plot
            self.update_line()
            i+=1

    def proc_cmd(self, cmd):
        time = 1
        for c in cmd.split(" "):
            if "D" in c:
                self.cmd_angle(c)
            elif "M" in c:
                self.cmd_speed(c)
            else:
                time = float(re.findall(r"\d*.\d*", c)[0])
        return time *1.25 
            

    def det_rotation(self, phi = None):
        """determines a direction vector derived from phi"""
        if phi is None: phi = self.phi 
        phi = phi / 360 * np.pi * 2
        dtheta = self.v*np.sin(phi)/self.L
        rotation_matrix = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])
        direction = np.matmul(rotation_matrix, self.direction)
        return direction

    def velocity(self, dt, f=None, decel=False,):
        if f is None: f = self.Famax
        temp0 = ((f / self.m) * np.square(dt))
        # if decel: temp0 = -((self.Fbmax / self.m) * np.square(dt))
        temp1 = ((self.calcdrag() / self.m) * np.square(dt))
        return self.v + temp0 - temp1

    def calcdrag(self) -> float:
        return (self.b * np.abs(self.v) + self.c * np.square(self.v))

    def cmd_angle(self, cmd):
        """D200 | D150 | D100
        turn a kit instruction into model values
        phi = 25 | phi = 0 | phi = -25
        LEFT | MID | RIGHT
        SETS STATE VALUE self.direction"""
        match cmd:
            case "D200":
                self.direction = self.det_rotation(25)
                self.phi = 25 
                return
            case "D170":
                self.direction = self.det_rotation(7)
                self.phi = 7 
                return
            case "D150":
                self.direction = self.det_rotation(0)
                self.phi = 0 
                return
            case "D130":
                self.direction = self.det_rotation(-11)
                self.phi = -11
                return
            case "D100":
                self.direction = self.det_rotation(-25)
                self.phi = -25 
                return
            
    def cmd_speed(self, cmd):
        """ 135 - 150: backwards
        150: standstill
        150-165: forwards
        SETS STATE VALUE self.velocity"""
        c = int(re.findall(r"\d{3}",cmd)[0])
        if (c < 150):
            self.f = -self.Fbmax
        else:
            match c:
                case 157:
                    f = self.m * 17.5
                case 158:
                    f = self.m * 20
                case 159:
                    f = self.m * 22.5
                case 160:
                    f = self.m * 25
                case 161:
                    f = self.m * 26.25
                case 162:
                    f = self.m * 27.5
                case 163:
                    f = self.m * 28.75
                case 164:
                    f = self.m * 30
                case _: f = None
            self.f = f

    def det_xy(self, dt):
        """
        x,y = x0 + dirx * v * dt, y0 + diry * v * dt"""
        return self.pos[0] + self.direction[0] * self.v * dt, self.pos[1] + self.direction[1] * self.v * dt
