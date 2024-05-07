import matplotlib.pyplot as plt
import numpy as np

df = np.loadtxt("C:/Users/Jesse/OneDrive/epo4/Plots/Data for showing time delay and 70ms cycle.csv", delimiter=',')

# For one axis
# plt.title("Delay estimation")
# plt.plot(df[100:115,2],df[100:115,0], label='Left sensor distance')
# plt.plot(df[100:115,2],df[100:115,1], label='Right sensor distance')
# plt.grid()
# plt.legend()
# plt.show()

# For two axes
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(df[100:115,2],df[100:115,0], label='Left sensor distance')
ax.plot(df[100:115,2],df[100:115,1], label='Right sensor distance')
ax2 = ax.twinx()
ax2.plot(df[100:115,2],df[100:115,3]*1000, label='Delay', color="C2", linestyle='--')
ax.legend(loc=0)
ax.grid()
ax.set_xlabel("System time [s]")
ax.set_ylabel(r"Distance [cm]")
ax2.set_ylabel(r"Delay [ms]")
ax2.legend(loc=1)
plt.title("Delay estimation")
plt.savefig('Delay estimation', bbox_inches="tight", dpi=600)
plt.show()

time = [i[3] for i in df]
print(np.average(time), 'average')