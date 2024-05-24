import time
import numpy as np
import matplotlib.pyplot as plt
from KITTMODEL import KITTMODEL
import regex as re

class TEST:
	def __init__(self) -> None:
		self.m = 5.6  # mass, [kg]
		self.b = 5  # viscous friction, [N m^-1 s]
		self.c = 0.1  # air drag, [N m^-2 s^2]
		self.Famax = 400  # Max acc force, [N]
		self.Fbmax = 500  # max brake force, [N]\
		self.L = 33.5  # Length of wheelbase


		self.figure = plt.figure()
		self.ax = self.figure.subplots()
		self.ax.grid()
		plt.show(block=False)

		# data
		self.positions = [(0,0)]

		# self.xy = [(0,0), (1,1), (2,2), (4,4), (6,7), (9,0)]

		self.lines, = self.ax.plot(*zip(*self.positions))

		# state variables
		self.v = 0
		self.a = 0
		self.phi = 0
		self.direction = [1, 0]
		self.t = 0

		

	def update_line(self):
		self.lines.set_data(*zip(*self.positions))
		self.ax.relim()
		self.ax.autoscale_view()
		self.figure.canvas.draw()
		self.figure.canvas.flush_events()
		
	def sim(self, inputs):
		i = 0
		for x in inputs:
			# process input
			self.proc_cmd(x)

			# simulate input
			

			# update plot
			self.update_line()
			i+=1

	def proc_cmd(self, cmd):
		for c in cmd.split(" "):
			if "D" in c:
				self.cmd_angle(c)
			elif "M" in c:
				self.cmd_speed(c)
			else:
				time = float(re.findall(r"\d*.\d*", c)[0])

			

	def det_rotation(self, phi = None):
		"""determines a direction vector derived from phi"""
		if phi is None: phi = self.phi
		dtheta = self.v*np.sin(phi)/self.L

		rotation_matrix = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])
		direction = np.matmul(rotation_matrix, self.direction)
		return direction

	def cmd_angle(self, cmd):
		"""D200 | D150 | D100
		turn a kit instruction into model values
		phi = 25 | phi = 0 | phi = -25
		LEFT | MID | RIGHT"""
		match cmd:
			case "D200":
				self.direction = self.det_rotation(25)
				return
			case "D150":
				self.direction = self.det_rotation(0)
				return
			case "D100":
				self.direction = self.det_rotation(-25)
				return
			
	def cmd_speed(self, cmd):
		self.v
		
    
commands= ["D150 M165 0.5", "M160", "D200", "M170", "M180", "D100", "M160", "M170"]
t= TEST()
t.sim(commands)
plt.show(block=True)
