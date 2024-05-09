import numpy as np
import matplotlib.pyplot as plt

class KITTMODEL():
    def __init__(self) -> None:
        self.m = 5.6 # mass, [kg]
        self.b = 5 # viscous friction, [N m^-1 s]
        self.c = 0.1 # air drag, [N m^-2 s^2]
        self.Famax = 400 # Max acc force, [N]
        self.Fbmax = 14 # max brake force, [N]

        #state params
        self.z = 0
        self.v = 0
        self.a = 0

    def velocity(self, dt, decel = False):
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
        t = np.arange(0,8.02,dt)
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

        t = np.arange(0,8.02,dt)
        ax = plt.figure().subplots(2)
        plt.title("Deceleration")
        vv, = ax[0].plot(t, velocity)
        ax[0].set_title("Velocity [km/h]")
        zz, = ax[1].plot(t, position)
        ax[1].set_title("Postion")

        plt.show()
        

if __name__ == "__main__":
    md = KITTMODEL()
    md.simulate()
    plt.show()
            


