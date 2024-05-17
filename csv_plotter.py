import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt

''''' file names: C:/Users/Jesse/OneDrive/epo4/Distancedata/Plots/
driving_to_wall
driving_from_wall
driving from and towards the wall
Data for showing time delay and 70ms cycle

more data: 
C:/Users/Jesse/OneDrive/epo4/Distancedata/
dtw_speed7 (8, 9, 10, 11_bty19,1, 12, 13, 14)
dtw_s=7,b=19,2
'''''
speed=14
df=np.loadtxt("C:/Users/Jesse/OneDrive/epo4/Distancedata/dtw_speed14.csv",delimiter=',')
# df = np.loadtxt("distance_data.csv", delimiter=',')

print(df)


def butter_lowpass(cutoff,fs,order=5):
    nyq=0.5 * fs
    normal_cutoff=cutoff / nyq
    bf,af=butter(order,normal_cutoff,btype='low',analog=False)
    return bf,af


order=3
fs=5
cutoff=0.3

b,a=butter_lowpass(cutoff,fs,order)

limit=-1
stepsize=4

filtered_distance_L=filtfilt(b,a,df[:limit:stepsize,0])
filtered_distance_R=filtfilt(b,a,df[:limit:stepsize,1])

# For one axis
fig,ax=plt.subplots(1,3,figsize=(10,6))
fig.suptitle(f'Driving to wall with speed = %d' % speed,fontsize=14,fontweight='bold')

ax[0].set_title("Distance estimation")
ax[0].plot(df[:limit:stepsize,2],filtered_distance_L,label='Left sensor distance')
ax[0].plot(df[:limit:stepsize,2],filtered_distance_R,label='Right sensor distance')
ax[0].grid()
ax[0].legend()

velocityL=-np.gradient(filtered_distance_L,df[:limit:stepsize,2])
velocityR=-np.gradient(filtered_distance_R,df[:limit:stepsize,2])
ax[1].set_title("Velocity estimation")
ax[1].plot(df[:limit:stepsize,2],velocityL,label='Left sensor distance')
ax[1].plot(df[:limit:stepsize,2],velocityR,label='Right sensor distance')
ax[1].grid()
ax[1].legend()

accelerationL=np.gradient(velocityL,df[:limit:stepsize,2])
accelerationR=np.gradient(velocityR,df[:limit:stepsize,2])
ax[2].set_title("Acceleration estimation")
ax[2].plot(df[:limit:stepsize,2],accelerationL,label='Left sensor distance')
ax[2].plot(df[:limit:stepsize,2],accelerationR,label='Right sensor distance')
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

time=[i[3] for i in df]
print(np.average(time),'average')
