
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
class self:
    #def __init__(self, recording, debug=False):
        # Store the recordings
        # Load the reference signal from memory
        #x_car, y_car = self.localization()
    

    def localization(self, audiowav):
        # Split each recording into individual pulses
        y11 = audiowav[:,0]
        y12 = audiowav[:,1]
        y13 = audiowav[:,2]
        y14 = audiowav[:,3]
        y15 = audiowav[:,4]

        # Calculate TDOA between different microphone pairs
        D12 = self.TDOA(y11, y12)
        D13 = self.TDOA(y11, y13)
        D14 = self.TDOA(y11, y14)
        
        # Run the coordinate_2d using the calculated TDOAs
        D12, D13, D14 = self.coordinate_2d(D12, D13, D14)



        return D12, D13, D14
        
    def TDOA(self, rec1, rec2, min_val=0.025):
        # Calculate channel estimation of each recording using ch2 or ch3
        h0 = self.ch3(rec1, refsignal)
        h1 = self.ch3(rec2, refsignal)
    
        # Calculate TDOA between two recordings based on peaks in the channel estimate
        start = 0
        for i, k in enumerate(h0):
            if np.abs(k) > min_val:
                print(f"found value above threshold; {i}")
                start = i

        # find peak of signal
        segmh0 = h0[start:start + 650]
        h0_peak = np.max(segmh0)
        h0_index = np.argmax(h0)

        # find h1 channel peak
        segmh1 = h1[start:start + 1000]
        h1_peak = np.max(segmh1)
        h1_index = np.argmax(h1)

        #return indices & max values
        print(f"peakh0: {np.abs(h0_peak)}, h0_index: {h0_index}")
        return (np.abs(h0_peak), h0_index, np.abs(h1_peak), h1_index)
        
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


if __name__ == "__main__":
    # Main block for testing
    # Read the .wav file
    # Localize the sound source
    # Present the results
    Fs_RX = 4000
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

    xyz_audiowav1_ch1, xyz_audiowav1_ch2, xyz_audiowav1_ch3, xyz_audiowav1_ch4, xyz_audiowav1_ch5 = self.localization(ABS1)
    xyz_audiowav2_ch1, xyz_audiowav2_ch2, xyz_audiowav2_ch3, xyz_audiowav2_ch4, xyz_audiowav2_ch5 = self.localization(ABS2)
    xyz_audiowav3_ch1, xyz_audiowav3_ch2, xyz_audiowav3_ch3, xyz_audiowav3_ch4, xyz_audiowav3_ch5 = self.localization(ABS3)
    xyz_audiowav4_ch1, xyz_audiowav4_ch2, xyz_audiowav4_ch3, xyz_audiowav4_ch4, xyz_audiowav4_ch5 = self.localization(ABS4)
    xyz_audiowav5_ch1, xyz_audiowav5_ch2, xyz_audiowav5_ch3, xyz_audiowav5_ch4, xyz_audiowav5_ch5 = self.localization(ABS5)
    xyz_audiowav6_ch1, xyz_audiowav6_ch2, xyz_audiowav6_ch3, xyz_audiowav6_ch4, xyz_audiowav6_ch5 = self.localization(ABS6)
    xyz_audiowav7_ch1, xyz_audiowav7_ch2, xyz_audiowav7_ch3, xyz_audiowav7_ch4, xyz_audiowav7_ch5 = self.localization(ABS7)
    xyz_audiowav8_ch1, xyz_audiowav8_ch2, xyz_audiowav8_ch3, xyz_audiowav8_ch4, xyz_audiowav8_ch5 = self.localization(ABS8)
    xyz_audiowav9_ch1, xyz_audiowav9_ch2, xyz_audiowav9_ch3, xyz_audiowav9_ch4, xyz_audiowav9_ch5 = self.localization(ABS9)
    xyz_audiowav10_ch1, xyz_audiowav10_ch2, xyz_audiowav10_ch3, xyz_audiowav10_ch4, xyz_audiowav10_ch5 = self.localization(ABS10)
    
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


    
    #PLOT #PLOT #PLOT #PLOT #PLOT #PLOT #PLOT #PLOT
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
    t = np.linspace(0, len(h11)*period,