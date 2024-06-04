import matplotlib.pyplot as plt
import numpy as np
import regex as re


class KITTMODEL:
    def __init__(self):
        self.m = 5.6  # mass, [kg]
        self.b = 5  # viscous friction, [N m^-1 s]
        self.c = 0.1  # air drag, [N m^-2 s^2]
        self.Famax = 400  # Max acc force, [N]
        self.Fbmax = 500  # max brake force, [N]
        self.L = 33.5  # Length of wheelbase

        # create figures & axes
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

        # initialise plots
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
        """updates the initialised figures with new data generated after running a command string"""
        self.lines.set_data(*zip(*self.positions))
        self.vellines.set_ydata(self.velocities)
        self.vellines.set_xdata(self.times)
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
        """simulates a list of command strings"""
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
                print(self.direction)
                self.pos = self.det_xy(self.dt)
                self.positions.append(self.pos)
                self.t += self.dt
                self.times.append(self.t)

            # update plot
            self.update_line()
            i+=1

    def proc_cmd(self, cmd):
        """process a single commandline e.g. \"D200 M160 2\"
        
        If no time is given, 1 second is used."""
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
        # convert phi to radians
        phi = phi / 360 * np.pi * 2
        
        # determine rotation matrix
        dtheta = self.v*np.sin(phi)/self.L
        rotation_matrix = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])
        
        direction = np.matmul(rotation_matrix, self.direction)
        return direction

    def velocity(self, dt, f=None):
        """calculates the new velocity of the car for time interval dt"""
        if f is None: f = self.Famax
        temp0 = ((f / self.m) * np.square(dt))
        # if decel: temp0 = -((self.Fbmax / self.m) * np.square(dt))
        temp1 = ((self.calcdrag() / self.m) * np.square(dt))
        return self.v + temp0 - temp1

    def calcdrag(self) -> float:
        """Determines the air drag the car experiences at its current velocity"""
        return (self.b * np.abs(self.v) + self.c * np.square(self.v))

    def cmd_angle(self, cmd):
        """D200 | D150 | D100

        turn a kit instruction into model values

        phi = 25 | phi = 0 | phi = -25

        LEFT | MID | RIGHT

        SETS STATE VALUE self.direction"""
        match cmd:
            case "D200":
                self.direction = self.det_rotation(19.4)
                self.phi = 19.4
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
                self.direction = self.det_rotation(-19.4)
                self.phi = -19.4
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
        """ determines x,y position of the KITT car following 
        x,y = x0 + dirx * v * dt, y0 + diry * v * dt"""
        return self.pos[0] + self.direction[0] * self.v * dt, self.pos[1] + self.direction[1] * self.v * dt

    def generate_curve_command(self, carloc, cart_rad, dest, threshold=0.01):
        """
        Generates the commands for the car to point in the direction of the destination.
        :param carloc: Location of the car
        :param cart_rad: Orientation of the car
        :param dest: Destination
        :param threshold: Amount of deviation the orientation of the car can have from the destination.
        :return: end position of the car, end direction (unit vector), direction command (100 or 200), time of movement
        """
        # Get the desired vector, and check if it lies on the left or right of the car.
        _, desired_rad, desired_vec = self.desired_vector(carloc, dest)
        print("Car rad", cart_rad, "Desired rad:", desired_rad, "Desired vec:", desired_vec)
        if desired_rad < cart_rad:
            print("Go right")
            dir_com = 100
            input_com = 'M157 D100 10'
        else:
            print("Go left")
            dir_com = 200
            input_com = 'M157 D200 10'

        # Process the command, and set the starting parameters
        time = self.proc_cmd(input_com)
        self.direction = (np.cos(cart_rad), np.sin(cart_rad))
        self.pos = carloc
        self.positions.clear()

        # Simulate the cars movement
        for t in np.arange(0, time, self.dt):
            # Calculates the difference between vector angles. If it lies within a treshold, stop simulating
            # and return the simulation values.
            diff = np.linalg.norm(np.subtract(desired_vec, self.direction))
            if diff > threshold:
                print("F:", self.f)
                self.v = self.velocity(self.dt, self.f)
                print("V: ", self.v)
                self.velocities.append(self.v)
                self.direction = self.det_rotation()
                self.pos = self.det_xy(self.dt)
                self.positions.append(self.pos)
                self.t += self.dt
                self.times.append(self.t)
                _, _, desired_vec = self.desired_vector(self.positions[-1], dest)
            else:
                print("Car is pointing to the destination! Car ran for:", t)
                # Plot the simulated curve
                plt.plot(*zip(*self.positions))
                plt.xlim(0,4)
                plt.ylim(0,4)
                plt.grid()
                plt.show()
                return self.positions[-1], self.direction, dir_com, round(t,3)
        # If the model cannot find a curved path to the destination, e.g. when it lies too close to the car, return -1
        print("No valid easy path found!")
        return None, None, None, None

    def desired_vector(self, carloc, dest):
        """
        Calculates the vector pointing from the car to the endpoint using its current location.
        :param carloc: Current location of the car
        :param dest: Destination
        :return: length (m), direction (rad), desired_vec (unit vector)
        """
        vector = [dest[0] - carloc[0],dest[1] - carloc[1]]
        length = np.sqrt(np.square(vector[0]) + np.square(vector[1]))
        direction = np.arctan2(vector[1], vector[0])
        # Returns the vectors length, direction in radii and direction as a unit vector.
        return length, direction, [np.cos(direction), np.sin(direction)]


if __name__ == "__main__":
    md = KITTMODEL()
    # inputs = ["D170 M160 1"]
    # md.sim(inputs)
    # plt.show(block=True)