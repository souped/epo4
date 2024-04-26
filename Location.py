
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse
#from IPython.display import Audio
from refsignal import refsignal            # model for the EPO4 audio beacon signal
from wavaudioread import wavaudioread
from recording_tool import recording_tool


#def __init__(self, recording, debug=False):
    # Store the recordings
    # Load the reference signal from memory
    # x_car, y_car = self.localization()
    
    
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
    
    fig, ax = plt.subplots(15, 1, figsize=(10,20))
    period = 1 / Fs_RX
    t = np.linspace(0, period*len(y1), len(y1))

    ## first plot
    ax[0].plot(t, y1, color='C0')
    ax[0].set_title("Recording X=64, Y=40, Channel 1")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Amplitude")

    ax[1].plot(t, y2, color='C0')
    ax[1].set_title("Recording X=64, Y=40, Channel 2")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Amplitude")

    ax[2].plot(t, y3, color='C0')
    ax[2].set_title("Recording X=64, Y=40, Channel 3")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Amplitude")

    ax[3].plot(t, y4, color='C0')
    ax[3].set_title("Recording X=64, Y=40, Channel 4")
    ax[3].set_xlabel("Time [s]")
    ax[3].set_ylabel("Amplitude")

    ax[4].plot(t, y5, color='C0')
    ax[4].set_title("Recording X=64, Y=40, Channel 5")
    ax[4].set_xlabel("Time [s]")
    ax[4].set_ylabel("Amplitude")

    t = np.linspace(0, len(h1)*period, len(h1))
    ## first plot
    ax[5].plot(t, h1, color='C0')
    ax[5].set_title("Estimation of recording")
    ax[5].set_xlabel("Time [s]")
    ax[5].set_ylabel("Amplitude")

    ax[6].plot(t, h2, color='C0')
    ax[6].set_title("Estimation of recording")
    ax[6].set_xlabel("Time [s]")
    ax[6].set_ylabel("Amplitude")

    ax[7].plot(t, h3, color='C0')
    ax[7].set_title("Estimation of recording")
    ax[7].set_xlabel("Time [s]")
    ax[7].set_ylabel("Amplitude")

    ax[8].plot(t, h4, color='C0')
    ax[8].set_title("Estimation of recording")
    ax[8].set_xlabel("Time [s]")
    ax[8].set_ylabel("Amplitude")

    ax[9].plot(t, h5, color='C0')
    ax[9].set_title("Estimation of recording")
    ax[9].set_xlabel("Time [s]")
    ax[9].set_ylabel("Amplitude")

    f = np.linspace(0, Fs_RX/1000, len(h1))
    ## first plot
    ax[10].plot(f, abs(H1), color='C0')
    ax[10].set_title("Frequency spectrum of channel estimation")
    ax[10].set_xlabel("Frequency [Hz]")
    ax[10].set_ylabel("Amplitude")
    ax[10].set_ylim(bottom=0)

    ax[11].plot(f, abs(H2), color='C0')
    ax[11].set_title("Frequency spectrum of channel estimation")
    ax[11].set_xlabel("Frequency [Hz]")
    ax[11].set_ylabel("Amplitude")
    ax[11].set_ylim(bottom=0)

    ax[12].plot(f, abs(H3), color='C0')
    ax[12].set_title("Frequency spectrum of channel estimation")
    ax[12].set_xlabel("Frequency [Hz]")
    ax[12].set_ylabel("Amplitude")
    ax[12].set_ylim(bottom=0)

    ax[13].plot(f, abs(H4), color='C0')
    ax[13].set_title("Frequency spectrum of channel estimation")
    ax[13].set_xlabel("Frequency [Hz]")
    ax[13].set_ylabel("Amplitude")
    ax[13].set_ylim(bottom=0)

    ax[14].plot(f, abs(H5), color='C0')
    ax[14].set_title("Frequency spectrum of channel estimation")
    ax[14].set_xlabel("Frequency [Hz]")
    ax[14].set_ylabel("Amplitude")
    ax[14].set_ylim(bottom=0)

    fig.tight_layout()
    plt.show()