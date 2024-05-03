import matplotlib.pyplot as plt
import numpy as np
df = np.loadtxt('distance_data.csv', delimiter=',')
plt.plot(df[:,2],df[:,0])
plt.plot(df[:,2],df[:,1])
plt.show()

time = [i[3] for i in df]
print(np.average(time), 'average')