
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

#def __init__(self, recording, debug=False):
    # Store the recordings
    # Load the reference signal from memory
    #x_car, y_car = self.localization()
    
    
#def localization(self):
    # Split each recording into individual pulses
    # Calculate TDOA between different microphone pairs
    # Run the coordinate_2d using the calculated TDOAs
    
#def TDOA(self, rec1, rec2):
    # Calculate channel estimation of each recording using ch2 or ch3
    # Calculate TDOA between two recordings based on peaks
    # in the channel estimate
    
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
    ABS2 = wavaudioread("opnames/record_x82_y399.wav", Fs_RX)
    ABS3 = wavaudioread("opnames/record_x109_y76.wav", Fs_RX)
    ABS4 = wavaudioread("opnames/record_x143_y296.wav", Fs_RX)
    ABS5 = wavaudioread("opnames/record_x150_y185.wav", Fs_RX)
    ABS6 = wavaudioread("opnames/record_x178_y439.wav", Fs_RX)
    ABS7 = wavaudioread("opnames/record_x232_y275.wav", Fs_RX)
    ABS8 = wavaudioread("opnames/record_x4_y_hidden_1.wav", Fs_RX)
    ABS9 = wavaudioread("opnames/record_x_y_hidden_2.wav", Fs_RX)
    ABS10 = wavaudioread("opnames/record_x_y_hidden_3.wav", Fs_RX)

    refsig = wavaudioread("opnames/reference.wav", Fs_RX)
    FTrefsig = fft(refsig)


    y11 = ABS1[:,0]
    y12 = ABS1[:,1]
    y13 = ABS1[:,2]
    y14 = ABS1[:,3]
    y15 = ABS1[:,4]

    h11 = ch3(refsig,y11)
    h12 = ch3(refsig,y12)
    h13 = ch3(refsig,y13)
    h14 = ch3(refsig,y14)
    h15 = ch3(refsig,y15)
    H11 = fft(h11)
    H12 = fft(h12)
    H13 = fft(h13)
    H14 = fft(h14)
    H15 = fft(h15)
    
    fig, ax = plt.subplots(3, 5, figsize=(20,10))
    period = 1 / Fs_RX
    t = np.linspace(0, period*len(y11), len(y11))

    ## first plot
    ax[0,0].plot(t, y11, color='C0')
    ax[0,0].set_title("Recording X=64, Y=40, Channel 1")
    ax[0,0].set_xlabel("Time [s]")
    ax[0,0].set_ylabel("Amplitude")

    ax[0,1].plot(t, y12, color='C0')
    ax[0,1].set_title("Recording X=64, Y=40, Channel 2")
    ax[0,1].set_xlabel("Time [s]")
    ax[0,1].set_ylabel("Amplitude")

    ax[0,2].plot(t, y13, color='C0')
    ax[0,2].set_title("Recording X=64, Y=40, Channel 3")
    ax[0,2].set_xlabel("Time [s]")
    ax[0,2].set_ylabel("Amplitude")

    ax[0,3].plot(t, y14, color='C0')
    ax[0,3].set_title("Recording X=64, Y=40, Channel 4")
    ax[0,3].set_xlabel("Time [s]")
    ax[0,3].set_ylabel("Amplitude")

    ax[0,4].plot(t, y15, color='C0')
    ax[0,4].set_title("Recording X=64, Y=40, Channel 5")
    ax[0,4].set_xlabel("Time [s]")
    ax[0,4].set_ylabel("Amplitude")

    t = np.linspace(0, len(h11)*period, len(h11))
    ## first plot
    ax[1,0].plot(t, h11, color='C0')
    ax[1,0].set_title("Estimation of recording")
    ax[1,0].set_xlabel("Time [s]")
    ax[1,0].set_ylabel("Amplitude")

    ax[1,1].plot(t, h12, color='C0')
    ax[1,1].set_title("Estimation of recording")
    ax[1,1].set_xlabel("Time [s]")
    ax[1,1].set_ylabel("Amplitude")

    ax[1,2].plot(t, h13, color='C0')
    ax[1,2].set_title("Estimation of recording")
    ax[1,2].set_xlabel("Time [s]")
    ax[1,2].set_ylabel("Amplitude")

    ax[1,3].plot(t, h14, color='C0')
    ax[1,3].set_title("Estimation of recording")
    ax[1,3].set_xlabel("Time [s]")
    ax[1,3].set_ylabel("Amplitude")

    ax[1,4].plot(t, h15, color='C0')
    ax[1,4].set_title("Estimation of recording")
    ax[1,4].set_xlabel("Time [s]")
    ax[1,4].set_ylabel("Amplitude")

    f = np.linspace(0, Fs_RX/1000, len(h11))
    ## first plot
    ax[2,0].plot(f, abs(H11), color='C0')
    ax[2,0].set_title("Frequency spectrum of channel estimation")
    ax[2,0].set_xlabel("Frequency [Hz]")
    ax[2,0].set_ylabel("Amplitude")
    ax[2,0].set_ylim(bottom=0)

    ax[2,1].plot(f, abs(H12), color='C0')
    ax[2,1].set_title("Frequency spectrum of channel estimation")
    ax[2,1].set_xlabel("Frequency [Hz]")
    ax[2,1].set_ylabel("Amplitude")
    ax[2,1].set_ylim(bottom=0)

    ax[2,2].plot(f, abs(H13), color='C0')
    ax[2,2].set_title("Frequency spectrum of channel estimation")
    ax[2,2].set_xlabel("Frequency [Hz]")
    ax[2,2].set_ylabel("Amplitude")
    ax[2,2].set_ylim(bottom=0)

    ax[2,3].plot(f, abs(H14), color='C0')
    ax[2,3].set_title("Frequency spectrum of channel estimation")
    ax[2,3].set_xlabel("Frequency [Hz]")
    ax[2,3].set_ylabel("Amplitude")
    ax[2,3].set_ylim(bottom=0)

    ax[2,4].plot(f, abs(H15), color='C0')
    ax[2,4].set_title("Frequency spectrum of channel estimation")
    ax[2,4].set_xlabel("Frequency [Hz]")
    ax[2,4].set_ylabel("Amplitude")
    ax[2,4].set_ylim(bottom=0)

    fig.tight_layout()
    plt.show()