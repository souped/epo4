
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

    #def coordinate_2d(self, D12, D13, D14):
        # Calculate 2D coordinates based on TDOA measurements
        # using the linear algebra given before
        
    def print_plots(a, refsig, Fs_RX, title, index):
        y11 = a[24000:30000,0]
        y12 = a[24000:30000,1]
        y13 = a[24000:30000,2]
        y14 = a[24000:30000,3]
        y15 = a[24000:30000,4]

        h11 = self.ch3(refsig,y11)
        h12 = self.ch3(refsig,y12)
        h13 = self.ch3(refsig,y13)
        h14 = self.ch3(refsig,y14)
        h15 = self.ch3(refsig,y15)
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
        ax[0,0].set_title("Recording Channel 1")
        ax[0,0].set_xlabel("Time [s]")
        ax[0,0].set_ylabel("Amplitude")

        ax[0,1].plot(t, y12, color='C0')
        ax[0,1].set_title("Recording Channel 2")
        ax[0,1].set_xlabel("Time [s]")
        ax[0,1].set_ylabel("Amplitude")

        ax[0,2].plot(t, y13, color='C0')
        ax[0,2].set_title("Recording Channel 3")
        ax[0,2].set_xlabel("Time [s]")
        ax[0,2].set_ylabel("Amplitude")

        ax[0,3].plot(t, y14, color='C0')
        ax[0,3].set_title("Recording Channel 4")
        ax[0,3].set_xlabel("Time [s]")
        ax[0,3].set_ylabel("Amplitude")

        ax[0,4].plot(t, y15, color='C0')
        ax[0,4].set_title("Recording Channel 5")
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
        ax[2,0].set_title("Frequency spectrum estimation")
        ax[2,0].set_xlabel("Frequency [Hz]")
        ax[2,0].set_ylabel("Amplitude")
        ax[2,0].set_ylim(bottom=0)

        ax[2,1].plot(f, abs(H12), color='C0')
        ax[2,1].set_title("Frequency spectrum estimation")
        ax[2,1].set_xlabel("Frequency [Hz]")
        ax[2,1].set_ylabel("Amplitude")
        ax[2,1].set_ylim(bottom=0)

        ax[2,2].plot(f, abs(H13), color='C0')
        ax[2,2].set_title("Frequency spectrum estimation")
        ax[2,2].set_xlabel("Frequency [Hz]")
        ax[2,2].set_ylabel("Amplitude")
        ax[2,2].set_ylim(bottom=0)

        ax[2,3].plot(f, abs(H14), color='C0')
        ax[2,3].set_title("Frequency spectrum estimation")
        ax[2,3].set_xlabel("Frequency [Hz]")
        ax[2,3].set_ylabel("Amplitude")
        ax[2,3].set_ylim(bottom=0)

        ax[2,4].plot(f, abs(H15), color='C0')
        ax[2,4].set_title("Frequency spectrum estimation")
        ax[2,4].set_xlabel("Frequency [Hz]")
        ax[2,4].set_ylabel("Amplitude")
        ax[2,4].set_ylim(bottom=0)

        plt.suptitle(title)
        fig.tight_layout()
        #plt.show()
        plt.savefig('plot_full_{}.png'.format(index), dpi=300)
        plt.close()
                

 

if __name__ == "__main__":
    # Main block for testing
    # Read the .wav file
    # Localize the sound source
    # Present the results
    Fs_RX = 48000
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
    #FTrefsig = fft(refsig)

    ABS = ["1","2","3","4","5","6","7","8","9","10"]
    
    for i in ABS:
        if i == "1":
            self.print_plots(ABS1, refsig, Fs_RX, "X = 64, Y = 40", 1)  
        """ if i == "2":
            self.print_plots(ABS2, refsig, Fs_RX, "X = 82, Y = 399", 2) 
        if i == "3":
            self.print_plots(ABS3, refsig, Fs_RX, "X = 109, Y = 76",3 ) 
        if i == "4":
            self.print_plots(ABS4, refsig, Fs_RX, "X = 143, Y = 296",4) 
        if i == "5":
            self.print_plots(ABS5, refsig, Fs_RX, "X = 150, Y = 185",5) 
        if i == "6":
            self.print_plots(ABS6, refsig, Fs_RX, "X = 178, Y = 439",6) 
        if i == "7":
            self.print_plots(ABS7, refsig, Fs_RX, "X = 232, Y = 275",7) 
        if i == "8":
            self.print_plots(ABS8, refsig, Fs_RX, "X = ?, Y = ?",8) 
        if i == "9":
            self.print_plots(ABS9, refsig, Fs_RX, "X = ?, Y = ?",9) 
        if i == "10":
            self.print_plots(ABS10, refsig, Fs_RX, "X = ?, Y = ?",10) 

 """