import numpy as np
import matplotlib.pyplot as plt


class KITTMODEL():
    def __init__(self) -> None:
        self.m = 5.6  # mass, [kg]
        self.b = 5  # viscous friction, [N m^-1 s]
        self.c = 0.1  # air drag, [N m^-2 s^2]
        self.Famax = 400  # Max acc force, [N]
        self.Fbmax = 500  # max brake force, [N]\
        self.L = 33.5  # Length of wheelbase

        # state params
        self.z = 0
        self.v = 0
        self.a = 0
        self.pos = [0, 0]
        self.direction = [0, 0]

        self.dt = 100

        # Assuming linearity
        speed1, force1 = 135, -self.Fbmax
        speed2, force2 = 165, self.Famax
        self.slope = (force2 - force1) / (speed2 - speed1)
        self.intercept = force1 - self.slope * speed1

    def determine_starting_angle(self):
        sp_string = input("Enter starting position:")
        sp_string = sp_string.split(" ")
        starting_pos = [int(sp_string[0]), int(sp_string[1])]
        self.pos = starting_pos
        if self.pos[0] == 0:
            self.direction = [1, 0]
        elif self.pos[1] == 0:
            self.direction = [0, 1]
        elif self.pos[0] == 480:
            self.direction = [0, -1]
        elif self.pos[1] == 480:
            self.direction = [-1, 0]
        else:
            print("Not a valid starting position! Try again.")
            self.determine_starting_angle()

    # This whole function is not correct.
    def change_rotation(self, theta, dt):
        phi = 0  # ???
        dtheta = self.v*np.sin(phi)/self.L

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        direction = np.matmul(rotation_matrix, self.direction)
        return direction

    def command_input(self):
        cmd_string = input("Enter commands:")
        # Example input:    D150 M165 0.5      Program just loops this function? Or give all commands in bulk?
        cmd_string = cmd_string.split(" ")
        speed = 0
        direction = 0
        time = 0
        for string in cmd_string:
            if "D" in string:
                print("Set direction:", string)
                direction = int(string)
            if "M" in string:
                print("Set speed:", string)
                speed = int(string)
            else:
                pass
        print("Time:", cmd_string[-1])

        positions = []
        velocities = []
        directions = []
        while time < float(cmd_string[-1]):
            self.v = self.change_velocity(self.speed_to_force_linear(speed), self.dt)
            velocities.append(self.v)
            self.direction = self.change_rotation(direction, self.dt)
            directions.append(self.direction)
            self.pos = self.change_position(self.dt)
            positions.append(self.pos)
            time += self.dt

    def change_position(self, dt):
        return self.pos + self.v * dt * self.direction

    def change_velocity(self, force, dt):
        temp0 = ((force / self.m) * np.square(dt))
        temp1 = ((self.calcdrag() / self.m) * np.square(dt))
        return self.v + temp0 - temp1

    def speed_to_force_linear(self, speed):
        force = self.slope * speed + self.intercept
        return force

    def velocity(self, dt, decel=False):
        temp0 = ((self.Famax / self.m) * np.square(dt))
        if decel: temp0 = -((self.Fbmax / self.m) * np.square(dt))
        temp1 = ((self.calcdrag() / self.m) * np.square(dt))
        return self.v + temp0 - temp1

    def calcdrag(self) -> float:
        return (self.b * np.abs(self.v) + self.c * np.square(self.v))

    def det_z(self, dt):
        """
        a = v'
        v = v0 + a*t
        z = z + v*t"""
        return self.z + self.v * dt

    def simulate(self, dt=0.01):
        time = 0
        velocity = [0]
        position = [0]
        while time < 8:
            self.v = self.velocity(dt)
            velocity.append(self.v)
            self.z = self.det_z(dt)
            position.append(self.z)
            time += dt

        # plot z, v
        t = np.arange(0, 8.02, dt)
        ax = plt.figure().subplots(2)
        plt.title("acceleration")
        vv, = ax[0].plot(t, velocity)
        ax[0].set_title("Velocity [m/s]")
        zz, = ax[1].plot(t, position)
        ax[1].set_title("Postion")

        # SIMULATING DECELERATION
        self.v = 4
        self.z = 0
        time = 0
        velocity = [4]
        position = [0]
        while time < 8:
            self.v = self.velocity(dt, True)
            velocity.append(self.v)
            self.z = self.det_z(dt)
            position.append(self.z)
            time += dt

        t = np.arange(0, 8.02, dt)
        ax = plt.figure().subplots(2)
        plt.title("Deceleration")
        vv, = ax[0].plot(t, velocity)
        ax[0].set_title("Velocity [km/h]")
        zz, = ax[1].plot(t, position)
        ax[1].set_title("Postion")

        plt.show()


if __name__ == "__main__":
    md = KITTMODEL()
    md.determine_starting_angle()
    md.command_input()
    md.simulate()
    plt.show()
