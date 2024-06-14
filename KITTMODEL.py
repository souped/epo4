import matplotlib.pyplot as plt
import numpy as np
import regex as re

from KITT_communication import KITT
from Keyboard import Keyboard

class KITTMODEL():
    def __init__(self) -> None:
        self.m = 5.6  # mass, [kg]
        self.b = 5  # viscous friction, [N m^-1 s]
        self.c = 0.1  # air drag, [N m^-2 s^2]
        self.Famax = 400  # Max acc force, [N]
        self.Fbmax = 500  # max brake force, [N]
        self.L = 33.5  # Length of wheelbase

        # create figures & axes
        self.plotting_enabled = 1
        # self.figure = plt.figure()
        # self.ax = self.figure.subplots()
        # self.figure2 = plt.figure()
        # self.ax2 = self.figure2.subplots()
        # self.ax.grid()

        # data
        self.positions = [(0,0)]
        self.velocities = [0]
        self.times = [0]
        self.modtime = 0

        # self.xy = [(0,0), (1,1), (2,2), (4,4), (6,7), (9,0)]

        # initialise plots, double # = (un)commented
        # self.lines, = self.ax.plot(*zip(*self.positions))
        # self.vellines, = self.ax2.plot(self.times, self.velocities)
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
        # plot speed
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
                self.v = self.velocity(self.dt, self.f)
                self.velocities.append(self.v)
                self.direction = self.det_rotation()
                self.pos = self.det_xy(self.dt)
                self.positions.append(self.pos)
                self.t += self.dt
                self.times.append(self.t)

                print(self.f, self.v)

            # update plot has been (un)commented!
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
        return time


    def det_rotation(self, phi = None):
        """determines a direction vector derived from phi"""
        if phi is None: phi = self.phi 
        # convert phi to radians
        phi = phi / 360 * np.pi * 2
        
        # determine rotation matrix
        dtheta = self.v*np.sin(phi)/self.L
        dtheta = dtheta*1.15
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
        """Determines the drag the car experiences at its current velocity"""
        return (self.b * np.abs(self.v) + self.c * np.square(self.v))

    def steering_angle(self, directionnr):
        return (directionnr / 50) * 19.4

    def cmd_angle(self, cmd):
        """D200 | D150 | D100

        turn a kit instruction into model values

        phi = 25 | phi = 0 | phi = -25

        LEFT | MID | RIGHT

        SETS STATE VALUE self.direction"""
        match cmd:
            case "D200":
                self.direction = self.det_rotation(19.4)
                self.phi = 19
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
                self.phi = -19
                return
            case _:
                pass

        #### ALTERNATIVE FOR EVERY DIRECTION?
        drnr = int(re.findall(r"\d{3}", cmd)[0])
        # drnr = math.floor(drnr)
        self.direction = self.det_rotation(self.steering_angle(drnr))
        self.phi = drnr
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

    def generate_straight_command(self, carloc, dest, threshold=0.05):
        """
        Generates the command to get the car to its destination,
        under the condition that it is pointing to the destination.
        :param carloc: Location of the car as (x,y)
        :param dest: Location of the destination as (x,y)
        :param threshold: Threshold the car can deviate from the destination.
        :return: Time the car needs to move.
        """
        # Set the starting parameters
        time = self.proc_cmd('M158 D150 10')
        # self.direction = (1,0)
        # self.pos = (0,0)
        # self.positions.clear()
        self.v = 0
        self.velocities.clear()
        self.t = 0
        # self.times.clear()

        # Calculate length to destination
        total_length, _, _ = self.desired_vector(carloc,dest)

        # Simulate the cars movement
        for t in np.arange(0, time, self.dt):
            length, _, _ = self.desired_vector(self.pos, dest)
            # print("Distance to destination:",length)
            if length >= threshold:
                # SETS THE FORCE TWICE THE MEASURED.
                # Since current forces work perfectly in corners, and the car goes twice as fast going straight.
                self.v = self.velocity(self.dt, self.f*1.75)
                self.velocities.append(self.v)
                self.pos = self.det_xy(self.dt)
                self.positions.append(self.pos)
                self.t += self.dt
                self.times.append(self.t)
            else:
                t = round(t,3)
                print("Simulated car is at the destination! Car ran for:", t)
                if self.plotting_enabled == 1:
                    self.plot_path(dest)
                self.modtime = t
                return f"M158 D150 {t}"
        return print("No path found!")

    def generate_curve_command(self, carloc, cart_rad, dest, dir_flip = 0):
        """
        Generates the commands for the car to point in the direction of the destination.
        :param carloc: Location of the car as (x,y)
        :param cart_rad: Orientation of the car in rad
        :param dest: Destination as (x,y)
        :return: end position of the car, end direction (unit vector), direction command (100 or 200), time of movement
        """
        def run_simulator(dir_flip):
            # Get the desired vector, and check if it lies on the left or right of the car.
            _, desired_rad, desired_vec = self.desired_vector(carloc, dest)
            if desired_rad < 0: desired_rad += 2*np.pi
            # print("Car rad", cart_rad, "Desired rad:", desired_rad, "Desired vec:", desired_vec)

            if desired_rad < cart_rad:
                if dir_flip == 0:
                    dir_com=100
                    print("Simulating when car goes right")
                else:
                    dir_com = 200
                    print("Simulating when car goes left")
            else:
                if dir_flip == 0:
                    dir_com = 200
                    print("Simulating when car goes left")
                else:
                    dir_com = 100
                    print("Simulating when car goes right")

            input_com=f'M158 D{dir_com} 10'
            return self.curve_command_simulator(input_com=input_com, carloc=carloc, cart_rad=cart_rad,
                                                desired_vec=desired_vec, dest=dest), dir_com

        (state, pos, dir, t), dir_com = run_simulator(dir_flip)

        if state == 1:
            return pos, dir, dir_com, t
        else:
            (state, pos, dir, t), dir_com = run_simulator(dir_flip=1)
            if state == 1:
                return pos, dir, dir_com, t
            else:
                return -1, 0, 0, 0

    def curve_command_simulator(self, input_com, carloc, cart_rad, desired_vec, dest, threshold=0.01):
        """
        Simulates the car's movement using the given parameters.
        :param input_com: The input command to be simulated
        :param carloc: Location of the car as (x,y) in m
        :param cart_rad: Direction of the car in rad
        :param desired_vec: Desired direction of the car in unit vector
        :param dest: Destination as (x,y) in m
        :param threshold: Threshold the car's orientation can deviate from the destination.
        :return: state, end position, end direction, time of movement.
        A None state means that the destination lies too close to the car, and a path cannot be formed from the car's current location.
        A '0' state means the created path goes out of bounds and a '1' state means the model successfully modeled a route.
        """
        print("Simulating...")
        time=self.proc_cmd(input_com)
        self.direction=(np.cos(cart_rad),np.sin(cart_rad))
        self.pos=carloc
        self.positions.clear()
        self.v=0
        self.velocities.clear()
        self.t=0
        self.times.clear()

        # Simulate the cars movement
        for t in np.arange(0,time,self.dt):
            # Calculates the difference between vector angles. If it lies within a treshold, stop simulating
            # and return the simulation values.
            diff=np.linalg.norm(np.subtract(desired_vec,self.direction))
            if diff > threshold:
                print("F:", self.f)
                if self.t < 1:
                    force = self.f * 0.3
                elif self.t < 2:
                    force = self.f * 0.7
                else:
                    force = self.f * 1.2

                self.v=self.velocity(self.dt,force)
                # print("V: ", self.v)
                self.velocities.append(self.v)
                self.direction=self.det_rotation()
                self.pos=self.det_xy(self.dt)
                self.positions.append(self.pos)
                self.t+=self.dt
                self.times.append(self.t)
                _,_,desired_vec=self.desired_vector(self.positions[-1],dest)
                if self.out_of_bounds(self.pos):
                    print("Simulated car is out of bounds!")
                    self.plot_path(dest)
                    return 0, 0, 0, 0
            else:
                t=round(t,3) * 0.8
                print("Simulated car is pointing to the destination! Car ran for:",t)
                self.modtime=t
                return 1,self.pos,self.direction,t

        # If the model cannot find a curved path to the destination, e.g. when it lies too close to the car, return -1
        print("No valid easy path found!")
        return None, None, None, None

    def out_of_bounds(self,pos):
        if 0 < pos[0] < 4.7 and 0 < pos[1] < 4.7:
            return False
        else:
            return True

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

    def plot_path(self, dest):
        # Plot the simulated curve
        fig, ax = plt.subplots()
        ax.plot(*zip(*self.positions))
        ax.set_xlim(0, 4.6)
        ax.set_ylim(0, 4.6)
        ax.set_aspect('equal')
        ax.plot(dest[0], dest[1], marker='o', color='red')
        plt.xlabel('X-axis [m]')
        plt.ylabel('Y-axis [m]')
        plt.grid()
        plt.show(block=False)


if __name__ == "__main__":
    md = KITTMODEL()
    #,"D100 M157 1", "D200 M160 2" "D200 M157 6.1",
    # inputs = ["150 M158 3"]
    # md.sim(inputs)
    # plt.show(block=True)
    dest = (1,2)
    kitt = KITT('COM5')
    xy, dir, dir_com, time = md.generate_curve_command(carloc=(0,0), cart_rad=0.5*np.pi, dest=dest)
    Keyboard.car_model_input(kitt,f"M158 D{dir_com} {time}")
    print(xy, time)
    # md.generate_straight_command(carloc=xy, dest=dest)
