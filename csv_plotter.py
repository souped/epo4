import matplotlib.pyplot as plt
import numpy as np

# file names:
# driving_to_wall
# driving_from_wall
# driving from and towards the wall
# Data for showing time delay and 70ms cycle

df = np.loadtxt("C:/Users/Jesse/OneDrive/epo4/Plots/driving_from_wall.csv", delimiter=',')
print(np.shape(df))
limit=64
# For one axis
fig, ax = plt.subplots(1,3, figsize = (10,6))

ax[0].set_title("Delay estimation")
ax[0].plot(df[:limit:4, 2], df[:limit:4, 0], label='Left sensor distance')
ax[0].plot(df[:limit:4, 2], df[:limit:4, 1], label='Right sensor distance')
ax[0].grid()
ax[0].legend()

velocityL = np.gradient(df[:limit:4, 0], df[:limit:4, 2])
velocityR = np.gradient(df[:limit:4, 1], df[:limit:4, 2])
ax[1].set_title("Velocity estimation")
ax[1].plot(df[:limit:4, 2], velocityL, label='Left sensor distance')
ax[1].plot(df[:limit:4, 2], velocityR, label='Right sensor distance')
ax[1].grid()
ax[1].legend()

accelerationL = np.gradient(velocityL, df[:limit:4, 2])
accelerationR = np.gradient(velocityR, df[:limit:4, 2])
ax[2].set_title("Acceleration estimation")
ax[2].plot(df[:limit:4, 2], accelerationL, label='Left sensor distance')
ax[2].plot(df[:limit:4, 2], accelerationR, label='Right sensor distance')
ax[2].grid()
ax[2].legend()

plt.tight_layout()
plt.show()
# # For two axes
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(df[100:115,2],df[100:115,0], label='Left sensor distance')
# ax.plot(df[100:115,2],df[100:115,1], label='Right sensor distance')
# ax2 = ax.twinx()
# ax2.plot(df[100:115,2],df[100:115,3]*1000, label='Delay', color="C2", linestyle='--')
# ax.legend(loc=0)
# ax.grid()
# ax.set_xlabel("System time [s]")
# ax.set_ylabel(r"Distance [cm]")
# ax2.set_ylabel(r"Delay [ms]")
# ax2.legend(loc=1)
# plt.title("Delay estimation")
# plt.savefig('Delay estimation', bbox_inches="tight", dpi=600)
# plt.show()

time = [i[3] for i in df]
print(np.average(time), 'average')
