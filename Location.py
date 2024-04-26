
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse
#from IPython.display import Audio
from refsignal import refsignal            # model for the EPO4 audio beacon signal
from wavaudioread import wavaudioread
from recording_tool import recording_tool


import numpy as np


y1 = y[:, 0]  # First channel
y2 = y[:, 1]  # Second channel
y3 = y[:, 2]  # Third channel
y4 = y[:, 3]  # Fourth channel
y5 = y[:, 4]  # Fifth channel








#def __init__(self, recording, debug=False):
    # Store the recordings
    # Load the reference signal from memory
    x_car, y_car = self.localization()
    
    
#def localization(self):
    # Split each recording into individual pulses
    # Calculate TDOA between different microphone pairs
    # Run the coordinate_2d using the calculated TDOAs
    
#def TDOA(self, rec1, rec2):
    # Calculate channel estimation of each recording using ch2 or ch3
    # Calculate TDOA between two recordings based on peaks
    # in the channel estimate
    @staticmethod
    
@staticmethod 
def ch3(x, y):
    Nx = len(x)           # Length of x
    Ny = len(y)             # Length of y
    L = Ny - Nx + 1          # Length of h
    Lhat = 1600000
    epsi = 0.001

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
    h = np.real(ifft(H))    
    #h = h[0:Lhat]      
    return h

def coordinate_2d(self, D12, D13, D14):
    # Calculate 2D coordinates based on TDOA measurements
    # using the linear algebra given before
    print()


if __name__ == "__main__":
    # Main block for testing
    # Read the .wav file
    # Localize the sound source
    # Present the results
    Fs_RX = 40000
    ABS1 = wavaudioread("opnames/record_x64_y40.wav", Fs_RX)

    refsig = wavaudioread("C:/Users/quint/Downloads/student_recording/student_recording/reference.wav", Fs_RX)
    FTrefsig = fft(refsig)


    y1 = ABS1[:,0]
    y2 = ABS1[:,1]
    y3 = ABS1[:,2]
    y4 = ABS1[:,3]
    y5 = ABS1[:,4]

    h1 = ch3(refsig,y1)
    h2 = ch3(refsig,y2)
    h3 = ch3(refsig,y3)
    h4 = ch3(refsig,y4)
    h5 = ch3(refsig,y5)
    H1 = fft(h1)
    H2 = fft(h2)
    H3 = fft(h3)
    H4 = fft(h4)
    H5 = fft(h5)
    
    fig, ax = plt.subplots(3, 5, figsize=(20,10))
    period = 1 / Fs_RX
    t = np.linspace(0, period*len(y1), len(y1))

    ## first plot
    ax[0,0].plot(t, y1, color='C0')
    ax[0,0].set_title("Recording X=64, Y=40, Channel 1")
    ax[0,0].set_xlabel("Time [s]")
    ax[0,0].set_ylabel("Amplitude")

    ax[0,1].plot(t, y2, color='C0')
    ax[0,1].set_title("Recording X=64, Y=40, Channel 2")
    ax[0,1].set_xlabel("Time [s]")
    ax[0,1].set_ylabel("Amplitude")

    ax[0,2].plot(t, y3, color='C0')
    ax[0,2].set_title("Recording X=64, Y=40, Channel 3")
    ax[0,2].set_xlabel("Time [s]")
    ax[0,2].set_ylabel("Amplitude")

    ax[0,3].plot(t, y4, color='C0')
    ax[0,3].set_title("Recording X=64, Y=40, Channel 4")
    ax[0,3].set_xlabel("Time [s]")
    ax[0,3].set_ylabel("Amplitude")

    ax[0,4].plot(t, y5, color='C0')
    ax[0,4].set_title("Recording X=64, Y=40, Channel 5")
    ax[0,4].set_xlabel("Time [s]")
    ax[0,4].set_ylabel("Amplitude")

    t = np.linspace(0, len(h1)*period, len(h1))
    ## first plot
    ax[1,0].plot(t, h1, color='C0')
    ax[1,0].set_title("Estimation of recording")
    ax[1,0].set_xlabel("Time [s]")
    ax[1,0].set_ylabel("Amplitude")

    ax[1,1].plot(t, h2, color='C0')
    ax[1,1].set_title("Estimation of recording")
    ax[1,1].set_xlabel("Time [s]")
    ax[1,1].set_ylabel("Amplitude")

    ax[1,2].plot(t, h3, color='C0')
    ax[1,2].set_title("Estimation of recording")
    ax[1,2].set_xlabel("Time [s]")
    ax[1,2].set_ylabel("Amplitude")

    ax[1,3].plot(t, h4, color='C0')
    ax[1,3].set_title("Estimation of recording")
    ax[1,3].set_xlabel("Time [s]")
    ax[1,3].set_ylabel("Amplitude")

    ax[1,4].plot(t, h5, color='C0')
    ax[1,4].set_title("Estimation of recording")
    ax[1,4].set_xlabel("Time [s]")
    ax[1,4].set_ylabel("Amplitude")

    f = np.linspace(0, Fs_RX/1000, len(h1))
    ## first plot
    ax[2,0].plot(f, abs(H1), color='C0')
    ax[2,0].set_title("Frequency spectrum of channel estimation")
    ax[2,0].set_xlabel("Frequency [Hz]")
    ax[2,0].set_ylabel("Amplitude")
    ax[2,0].set_ylim(bottom=0)

    ax[2,1].plot(f, abs(H2), color='C0')
    ax[2,1].set_title("Frequency spectrum of channel estimation")
    ax[2,1].set_xlabel("Frequency [Hz]")
    ax[2,1].set_ylabel("Amplitude")
    ax[2,1].set_ylim(bottom=0)

    ax[2,2].plot(f, abs(H3), color='C0')
    ax[2,2].set_title("Frequency spectrum of channel estimation")
    ax[2,2].set_xlabel("Frequency [Hz]")
    ax[2,2].set_ylabel("Amplitude")
    ax[2,2].set_ylim(bottom=0)

    ax[2,3].plot(f, abs(H4), color='C0')
    ax[2,3].set_title("Frequency spectrum of channel estimation")
    ax[2,3].set_xlabel("Frequency [Hz]")
    ax[2,3].set_ylabel("Amplitude")
    ax[2,3].set_ylim(bottom=0)

    ax[2,4].plot(f, abs(H5), color='C0')
    ax[2,4].set_title("Frequency spectrum of channel estimation")
    ax[2,4].set_xlabel("Frequency [Hz]")
    ax[2,4].set_ylabel("Amplitude")
    ax[2,4].set_ylim(bottom=0)

    fig.tight_layout()
    plt.show()