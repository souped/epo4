from wavaudioread import wavaudioread
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse

Fs_RX = 40000
ABS1 = wavaudioread("C:/Users/quint/Downloads/student_recording/student_recording/record_x64_y40.wav", Fs_RX)
#ABS2 = wavaudioread("student_recording/Recording-5-50cm.wav", Fs_RX)
#ABS3 = wavaudioread("student_recording/Recording-5-1m.wav", Fs_RX)

#########Plotting transmit sequence and its spectrum
refsig = wavaudioread("C:/Users/quint/Downloads/student_recording/student_recording/reference.wav", Fs_RX)
FTrefsig = fft(refsig)

#fig, ax = plt.subplots(1, 2, figsize=(20, 5))
#period = 1/Fs_RX
#t = np.linspace(0, 600*period, 600)
#f = np.linspace(0, Fs_RX/1000, len(refsig))

#ax[0].plot(t, refsig[0:600], color='C4')
#ax[0].set_title("Time-amplitude plot of transmitted signal")
#ax[0].set_xlabel("Time [s]")
#ax[0].set_ylabel("Amplitude")

#ax[1].plot(f[:len(refsig )], abs(FTrefsig), color = 'C4')
#ax[1].set_title("Frequency spectrum of transmitted signal")
#ax[1].set_xlabel("Frequency [kHz]")
#ax[1].set_ylabel("Amplitude")
#ax[1].set_ylim(bottom=0)

#plt.tight_layout()
#############################################Recieved sound
# Initialize the subplots
fig, ax = plt.subplots(1, 2, figsize=(20,5))

# Calculate the period
period = 1 / Fs_RX

t = np.linspace(0, period*3000, 3000)
## first plot
ax[0].plot(t, ABS1[25000:28000,0], color='C0')
ax[0].set_title("Recording at 1cm distance")
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Amplitude")

#t = np.linspace(0, 3000*period, 3000)
## Second plot
#ax[1].plot(t, ABS2[25500:28500,0], color='C1')
#ax[1].set_title("Recording at 50cm distance")
#ax[1].set_xlabel("Time [s]")
#ax[1].set_ylabel("Amplitude")

#t = np.linspace(0, 3000*period, 3000)
## Third plot
#ax[2].plot(t, ABS3[28500:31500,0], color='C2')
#ax[2].set_title("Recording at 1m distance")
#ax[2].set_xlabel("Time [s]")
#ax[2].set_ylabel("Amplitude")

#plt.tight_layout()

def ch3(x,y,Lhat,epsi):
    Nx = len(x)           # Length of x
    Ny = len(y)             # Length of y
    L = Ny - Nx + 1          # Length of h

    # Force x to be the same length as y
    x = np.append(x, [0]* (L-1))     # Make x same length as y

    # Deconvolution in frequency domain
    Y = fft(y)
    X = fft(x)

    # Threshold to avoid blow ups of noise during inversion
    ii = np.abs(X) < epsi*np.max(np.abs(X))
    X=X[:len(Y)]
    Y=Y[:len(X)]
    H = np.divide(Y,X)
    H = [0 if condition else x for x, condition in zip(Y, ii)]
    h = np.real(ifft(H))    # ensure the result is real
    h = h[0:Lhat]      # optional: truncate to length Lhat (L is not reliable?)
    return h

# Channel estimation via ch3
# suitable epsi: try values between 0.001 and 0.05
epsi = 0.001
Lhat = 3000
h1 = ch3(refsig,ABS1[25000:28000,0],Lhat,epsi)
#h2 = ch3(refsig,ABS2[25500:28500,0],Lhat,epsi)
#h3 = ch3(refsig,ABS3[28500:31500,0],Lhat,epsi)

#####################################################Reconstructed signals
# Initialize the subplots
fig, ax = plt.subplots(2, 2, figsize=(20, 10))

# Calculate the period
period = 1 / Fs_RX

t = np.linspace(0, len(h1)*period, len(h1))
## first plot
ax[0,0].plot(t, h1, color='C0')
ax[0,0].set_title("Channel estimation of recording at 1cm distance")
ax[0,0].set_xlabel("Time [s]")
ax[0,0].set_ylabel("Amplitude")

#t = np.linspace(0, len(h2)*period, len(h2))
## Second plot
## your code here ##
#ax[0,1].plot(t[:len(h2 )], h2, color='C1')
#ax[0,1].set_title("Channel estimation of recording at 50cm distance")
#ax[0,1].set_xlabel("Time [s]")
#ax[0,1].set_ylabel("Amplitude")

#t = np.linspace(0, len(h3)*period, len(h3))
## Third plot
## your code here ##
#ax[0,2].plot(t[:len(h3 )], h3, color='C2')
#ax[0,2].set_title("Channel estimation of recording at 1m distance")
#ax[0,2].set_xlabel("Time [s]")
#ax[0,2].set_ylabel("Amplitude")


f = np.linspace(0, Fs_RX/1000, len(h1))
H1 = fft(h1)
## first plot
ax[1,0].plot(f, abs(H1), color='C0')
ax[1,0].set_title("Frequency spectrum of channel estimation at 1cm distance")
ax[1,0].set_xlabel("Frequency [Hz]")
ax[1,0].set_ylabel("Amplitude")
ax[1,0].set_ylim(bottom=0)

#f = np.linspace(0, Fs_RX/1000, len(h2))
#H2 = fft(h2)
## Second plot
## your code here ##
#ax[1,1].plot(f, abs(H2), color='C1')
#ax[1,1].set_title("Frequency spectrum of channel estimation at 50cm distance")
#ax[1,1].set_xlabel("Frequency [Hz]")
#ax[1,1].set_ylabel("Amplitude")
#ax[1,1].set_ylim(bottom=0)

#f = np.linspace(0, Fs_RX/1000, len(h3))
#H3 = fft(h3)
## Third plot
## your code here ##
#ax[1,2].plot(f, abs(H3), color='C2')
#ax[1,2].set_title("Frequency spectrum of channel estimation at 1m distance")
#ax[1,2].set_xlabel("Frequency [Hz]")
#ax[1,2].set_ylabel("Amplitude")
#ax[1,2].set_ylim(bottom=0)

#plt.tight_layout()
plt.show()