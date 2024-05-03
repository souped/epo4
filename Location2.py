
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse
#from IPython.display import Audio
from refsignal import refsignal            # model for the EPO4 audio beacon signal
from wavaudioread import wavaudioread
from recording_tool import recording_tool
from sympy import symbols, solve




import numpy as np
class localization:
    #def __init__(recording, debug=False):
        # Store the recordings
        #x_car, y_car = self.localization()
        
        

    def localization(audiowav):
        # Split each recording into individual pulses
        y11 = audiowav[:,0]
        y12 = audiowav[:,1]
        y13 = audiowav[:,2]
        y14 = audiowav[:,3]
        y15 = audiowav[:,4]

        # Calculate TDOA between different microphone pairs
        D14 = localization.TDOA(y11, y14)
        
        
       
    
        
        
        # Run the coordinate_2d using the calculated TDOAs
        x, y = localization.coordinate_2d(D14, D23, D12, D43)
        return x, y
        
    def TDOA(rec1, rec2, min_val=400):
        
        # Calculate channel estimation of each recording using ch2 or ch3
        refsig = wavaudioread("opnames/reference.wav", Fs_RX)

        h0 = localization.ch3(rec1, refsig)
        h1 = localization.ch3(rec2, refsig)
        
    
        # Calculate TDOA between two recordings based on peaks in the channel estimate
        start = 0
        for i, k in enumerate(h0):
            if np.abs(k) > min_val:
                start = i
                break
        
    # find peak of signal
        print(start)
        segmh0 = h0[start:start+200000]
        h0_peak = np.max(segmh0)
        h0_index = np.argmax(segmh0)
        # find h1 channel peak
        segmh1 = h1[start:start+200000]
        h1_peak = np.max(segmh1)
        h1_index = np.argmax(segmh1)

        #return indices & max values
        print(f"peakh0: {np.abs(h0_peak)}, h1_peak: {np.abs(h1_peak)}")
        print(f"index0: {np.abs(h0_index)}, index1: {np.abs(h1_index)}")
        D=((h1_index - h0_index) / Fs_RX * 343)

        print(f"Time difference between peaks: {(h1_index - h0_index) / Fs_RX} s")
        print(f"Distance (referenced to 0): {D} m\n\n")
            
        fig, ax = plt.subplots(2, 1, figsize=(20,10))
        #period = 1 / Fs_RX
        #t = np.linspace(0, period*len(rec1), len(rec1))
        #
        ax[0].plot(h1_index, h1_peak, color='green', marker='o', markersize=10, label="Detected Peak")
        ax[0].plot(h1, color='C0')
        ax[0].set_title("Estimation of recording")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel("Amplitude")
        
        ax[1].plot(h0_index, h0_peak, color='green', marker='o', markersize=10, label="Detected Peak")
        ax[1].plot(h0, color='C0')
        ax[1].set_title("Estimation of recording")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Amplitude")
        
        fig.tight_layout()
        #plt.show()
        plt.savefig('plot_full_.png', dpi=300)
        plt.close()
        
        return D
        
    @staticmethod 
    def ch3(sig2, sig1):
        Nsig1 = len(sig1)           # Length of x
        Nsig2 = len(sig2)             # Length of y
        L = Nsig2 - Nsig1 + 1          # Length of h
        Lhat = 1600000
        epsi = 0.001

        # Force x to be the same length as y
        sig1 = np.append(sig1, [0]* (L-1))     # Make x same length as y

        # Deconvolution in frequency domain
        SIG2 = fft(sig2)
        SIG1 = fft(sig1)

        # Threshold to avoid blow ups of noise during inversion
        ii = np.abs(SIG1) < epsi*np.max(np.abs(SIG1))
        SIG1=SIG1[:len(SIG2)]
        SIG2=SIG2[:len(SIG1)]
        H = np.divide(SIG2,SIG1)
        H = [0 if condition else sig1 for sig1, condition in zip(SIG2, ii)]
        h = np.real(ifft(H))    
        #h = h[0:Lhat]      
        return h
    
    def coordinate_2d(D14, D23, D12, D43):
        # Calculate 2D coordinates based on TDOA measurements
        mic1 = [0, 0, 50]
        mic2 = [0, 480, 50]
        mic3 = [480, 480, 50]
        mic4 = [480, 0, 50]
        mic5 = [0, 240, 80]
        x, y = symbols('x, y')	
        
        cirkel1 = (x - mic1[0])**2 + (y - mic1[1])**2 - D14**2
        cirkel2 = (x - mic2[0])**2 + (y - mic2[1])**2 - D23**2
        cirkel3 = (x - mic1[0])**2 + (y - mic1[1])**2 - D12**2
        cirkel4 = (x - mic4[0])**2 + (y - mic4[1])**2 - D43**2
        # using the linear algebra given before
        snijpunten = solve((cirkel1, cirkel2, cirkel3, cirkel4), (x, y))

        return (snijpunten)
    
    def print_plots(a, refsig, Fs_RX, title, index, h1_index, h1_peak, h0_index, h0_peak):
        y11 = a[20000:40000,0]
        y12 = a[20000:40000,1]
        y13 = a[20000:40000,2]
        y14 = a[20000:40000,3]
        y15 = a[20000:40000,4]

        h11 = localization.ch3(refsig,y11)
        h12 = localization.ch3(refsig,y12)
        h13 = localization.ch3(refsig,y13)
        h14 = localization.ch3(refsig,y14)
        h15 = localization.ch3(refsig,y15)
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
        ax[1,0].plot(h0_index, h0_peak, color='green', marker='o', markersize=10, label="Detected Peak")
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
        ax[1,3].plot(h1_index, h1_peak, color='green', marker='o', markersize=10, label="Detected Peak")
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
    localizer = localization()
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
    
    x_car1, y_car1 = localization.localization(ABS1)
    #x_car2, y_car2 = localization.localization(ABS2)
    #x_car3, y_car3 = localization.localization(ABS3)
    #x_car4, y_car4 = localization.localization(ABS4)
    #x_car5, y_car5 = localization.localization(ABS5)
    #x_car6, y_car6 = localization.localization(ABS6)
    #x_car7, y_car7 = localization.localization(ABS7)
    #x_car8, y_car8 = localization.localization(ABS8)
    #x_car9, y_car9 = localization.localization(ABS9)
    #x_car10, y_car10 = localization.localization(ABS10)
    
    
    
    print(x_car1, y_car1)
    #print(x_car2, y_car2)
    #print(x_car3, y_car3)
    #print(x_car4, y_car4)
    #print(x_car5, y_car5)
    #print(x_car6, y_car6)
    #print(x_car7, y_car7)
    #print(x_car8, y_car8)
    #print(x_car9, y_car9)
    #print(x_car10, y_car10)
    
        