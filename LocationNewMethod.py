
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse, find_peaks
#from IPython.display import Audio
from refsignal import refsignal            # model for the EPO4 audio beacon signal
from wavaudioread import wavaudioread
from recording_tool import recording_tool
from sympy import symbols, solve
import numpy as np
import math



class localization:
    #def __init__(recording, debug=False):
        # Store the recordings
        #x_car, y_car = self.localization()
        
        

    def localization(audiowav):
        # Split each recording into individual pulses
        audio_channel1 = audiowav[:,0]
        audio_channel2 = audiowav[:,1]
        audio_channel3 = audiowav[:,2]
        audio_channel4 = audiowav[:,3]
        audio_channel5 = audiowav[:,4]
        Fref, ref_signal = wavfile.read("opnames/reference.wav")
        ref_signal =  ref_signal[:,0]
        
        
        #segments
        segments_channel1 = localization.detect_segments(audio_channel1)
        segments_channel2 = localization.detect_segments(audio_channel2)
        segments_channel3 = localization.detect_segments(audio_channel3)
        segments_channel4 = localization.detect_segments(audio_channel4)
        segments_channel5 = localization.detect_segments(audio_channel5)
        refsig = localization.detect_segments(ref_signal)
        ref = refsig[12]
        ref = ref[750:1500]

        # plt.figure(figsize=(10,5))
        # plt.plot(audio_channel1, color='C0')
        # plt.title("Input audio")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.show()

        # plt.figure(figsize=(10,5))
        # plt.plot(abs(segments_channel1[5]), color='C0')
        # plt.title("Input segment 5")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.show()

        # plt.figure(figsize=(10,5))
        # plt.plot(abs(ref_signal), color='C0')
        # plt.title("Ref signal")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.show()

        # plt.figure(figsize=(10,5))
        # plt.plot(abs(ref), color='C0')
        # plt.title("Ref signal small part")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.show()

        
        #channel estimation
        channel_responses_1 = [localization.ch3(segment, ref) for segment in segments_channel1]
        channel_responses_array_1 = np.array(channel_responses_1)
        channel_responses_2 = [localization.ch3(segment, ref) for segment in segments_channel2]
        channel_responses_array_2 = np.array(channel_responses_2)
        channel_responses_3 = [localization.ch3(segment, ref) for segment in segments_channel3]
        channel_responses_array_3 = np.array(channel_responses_3)
        channel_responses_4 = [localization.ch3(segment, ref) for segment in segments_channel4]
        channel_responses_array_4 = np.array(channel_responses_4)
        channel_responses_5 = [localization.ch3(segment, ref) for segment in segments_channel5]
        channel_responses_array_5 = np.array(channel_responses_5)
        
        
        #peaks
        peaks_channel1 = localization.find_segment_peaks(channel_responses_array_1)
        peaks_channel2 = localization.find_segment_peaks(channel_responses_array_2)
        peaks_channel3 = localization.find_segment_peaks(channel_responses_array_3)
        peaks_channel4 = localization.find_segment_peaks(channel_responses_array_4)
        peaks_channel5 = localization.find_segment_peaks(channel_responses_array_5) 
        
        print(peaks_channel1)
        print(peaks_channel2)
        print(peaks_channel3)
        print(peaks_channel4)
        print(peaks_channel5)

        sorted_peaks_1 = np.sort(peaks_channel1)
        trimmed_peaks_1 = sorted_peaks_1[10:-10]
        mean_peak_1 = np.mean(trimmed_peaks_1)
        
        sorted_peaks_2 = np.sort(peaks_channel2)
        trimmed_peaks_2 = sorted_peaks_2[10:-10]
        mean_peak_2 = np.mean(trimmed_peaks_2)
        
        sorted_peaks_3 = np.sort(peaks_channel3)
        trimmed_peaks_3 = sorted_peaks_3[10:-10]
        mean_peak_3 = np.mean(trimmed_peaks_3)

        sorted_peaks_4 = np.sort(peaks_channel4)
        trimmed_peaks_4 = sorted_peaks_4[10:-10]
        mean_peak_4 = np.mean(trimmed_peaks_4)
        
        sorted_peaks_5 = np.sort(peaks_channel5)
        trimmed_peaks_5 = sorted_peaks_5[10:-10]
        mean_peak_5 = np.mean(trimmed_peaks_5)

        # Calculate TDOA between different microphone pairs
        TDOA12 = localization.TDOA(mean_peak_1, mean_peak_2)
        TDOA13 = localization.TDOA(mean_peak_1, mean_peak_3)
        TDOA14 = localization.TDOA(mean_peak_1, mean_peak_4)
        TDOA23 = localization.TDOA(mean_peak_2, mean_peak_3)
        TDOA24 = localization.TDOA(mean_peak_2, mean_peak_4)
        TDOA34 = localization.TDOA(mean_peak_3, mean_peak_4)
        
        print(TDOA12)
        print(TDOA13)
        print(TDOA14)
        
        x, y = localization.coordinate_2d(TDOA12, TDOA13, TDOA14, TDOA23, TDOA24, TDOA34)
        
        return x, y
    
    def detect_segments(audio_signal):
        segments = []
        num_segments = 40
        segment_length = len(audio_signal) // num_segments
        segments = [audio_signal[i*segment_length : (i+1)*segment_length] for i in range(num_segments)]
        return segments    

    def find_segment_peaks(segment_signal):
        peaks_list = []
        for segment in segment_signal:
            peaks = np.argmax(segment)
            peaks_list.append(peaks)
              
        return peaks_list
    
    def TDOA(peak1, peak2):
        mean_tdoa = (peak2 - peak1)/Fs * 343
        return mean_tdoa

            
        
                
    @staticmethod 
    def ch3(signal_1, reference_signal):
        Nsignal_1 = len(signal_1)           # Length of x
        Nreference_signal = len(reference_signal)             # Length of y
        L = Nsignal_1 - Nreference_signal + 1          # Length of h
        Lhat = max(len(reference_signal), len(signal_1)) 
        epsi = 0.005

        # Force x to be the same length as y
        reference_signal = np.append(reference_signal, [0]* (L-1))     # Make x same length as y
        print(len(reference_signal))



        # Deconvolution in frequency domain
        fft_signal_1 = fft(signal_1)
        fft_reference_signal = fft(reference_signal)

        # Threshold to avoid blow ups of noise during inversion
        ii = (np.abs(fft_reference_signal)) < (np.max(np.abs(fft_reference_signal))*epsi)

        H = np.divide(fft_signal_1, fft_reference_signal)
        H[ii] = 0

        #H = [0 if condition else reference_signal for reference_signal, condition in zip(fft_signal_1, ii)]
        h = np.real(ifft(H))    
        h = h[0:Lhat]

        return abs(h)

    def coordinate_2d(D12, D13, D14, D23, D24, D34):
        
        D23= D13-D12
        D24= D14-D12
        D34= 14-D13
        
        # Calculate 2D coordinates based on TDOA measurements
        X1 = np.array([0, 0])
        X2 = np.array([0, 4.8])
        X3 = np.array([4.8, 4.8])
        X4 = np.array([4.8, 0])
        X5 = np.array([0, 2.4])
        
        norm_X1 = np.linalg.norm(X1)
        norm_X2 = np.linalg.norm(X2)
        norm_X3 = np.linalg.norm(X3)
        norm_X4 = np.linalg.norm(X4)
       
        
        B = np.array([
        [D12**2 - norm_X1**2 + norm_X2**2],
        [D13**2 - norm_X1**2 + norm_X3**2],
        [D14**2 - norm_X1**2 + norm_X4**2],
        [D23**2 - norm_X2**2 + norm_X3**2],
        [D24**2 - norm_X2**2 + norm_X4**2],
        [D34**2 - norm_X3**2 + norm_X4**2]
        ])
        
        X21=(X2-X1)
        X31=(X3-X1)
        X41=(X4-X1)
        X32=(X3-X2)
        X42=(X4-X2)
        X43=(X4-X3)
        

        
        A = np.array([
        [2*X21[0], 2*X21[1], -2 * D12, 0, 0],
        [2*X31[0], 2*X31[1], 0, -2 * D13, 0],
        [2*X41[0], 2*X41[1], 0, 0, -2 * D14],
        [2*X32[0], 2*X32[1], 0, -2 * D23, 0],
        [2*X42[0], 2*X42[1], 0, 0, -2 * D24],
        [2*X43[0], 2*X43[1], 0, 0, -2 * D34]
        ])
    
        
        
      
        y=np.linalg.pinv(A) @ B
        print(y)
        coordinate = y[:2]
        print(coordinate)
                

        

        return x, y

    # def print_plots(a, refsig, Fs_RX, title, index, h1_index, h1_peak, h0_index, h0_peak):
    #     y11 = a[20000:40000,0]
    #     y12 = a[20000:40000,1]
    #     y13 = a[20000:40000,2]
    #     y14 = a[20000:40000,3]
    #     y15 = a[20000:40000,4]

    #     h11 = localization.ch3(refsig,y11)
    #     h12 = localization.ch3(refsig,y12)
    #     h13 = localization.ch3(refsig,y13)
    #     h14 = localization.ch3(refsig,y14)
    #     h15 = localization.ch3(refsig,y15)
    #     H11 = fft(h11)
    #     H12 = fft(h12)
    #     H13 = fft(h13)
    #     H14 = fft(h14)
    #     H15 = fft(h15)

    #     #PLOT #PLOT #PLOT #PLOT #PLOT #PLOT #PLOT #PLOT

    #     fig, ax = plt.subplots(3, 5, figsize=(20,10))
    #     period = 1 / Fs_RX
    #     t = np.linspace(0, period*len(y11), len(y11))

    #     ## first plot
    #     ax[0,0].plot(t, y11, color='C0')
    #     ax[0,0].set_title("Recording Channel 1")
    #     ax[0,0].set_xlabel("Time [s]")
    #     ax[0,0].set_ylabel("Amplitude")

    #     ax[0,1].plot(t, y12, color='C0')
    #     ax[0,1].set_title("Recording Channel 2")
    #     ax[0,1].set_xlabel("Time [s]")
    #     ax[0,1].set_ylabel("Amplitude")

    #     ax[0,2].plot(t, y13, color='C0')
    #     ax[0,2].set_title("Recording Channel 3")
    #     ax[0,2].set_xlabel("Time [s]")
    #     ax[0,2].set_ylabel("Amplitude")

    #     ax[0,3].plot(t, y14, color='C0')
    #     ax[0,3].set_title("Recording Channel 4")
    #     ax[0,3].set_xlabel("Time [s]")
    #     ax[0,3].set_ylabel("Amplitude")

    #     ax[0,4].plot(t, y15, color='C0')
    #     ax[0,4].set_title("Recording Channel 5")
    #     ax[0,4].set_xlabel("Time [s]")
    #     ax[0,4].set_ylabel("Amplitude")

    #     t = np.linspace(0, len(h11)*period, len(h11))
    #     ## first plot
    #     ax[1,0].plot(t, h11, color='C0')
    #     ax[1,0].set_title("Estimation of recording")
    #     ax[1,0].plot(h0_index, h0_peak, color='green', marker='o', markersize=10, label="Detected Peak")
    #     ax[1,0].set_xlabel("Time [s]")
    #     ax[1,0].set_ylabel("Amplitude")

    #     ax[1,1].plot(t, h12, color='C0')
    #     ax[1,1].set_title("Estimation of recording")
    #     ax[1,1].set_xlabel("Time [s]")
    #     ax[1,1].set_ylabel("Amplitude")

    #     ax[1,2].plot(t, h13, color='C0')
    #     ax[1,2].set_title("Estimation of recording")
    #     ax[1,2].set_xlabel("Time [s]")
    #     ax[1,2].set_ylabel("Amplitude")

    #     ax[1,3].plot(t, h14, color='C0')
    #     ax[1,3].set_title("Estimation of recording")
    #     ax[1,3].plot(h1_index, h1_peak, color='green', marker='o', markersize=10, label="Detected Peak")
    #     ax[1,3].set_xlabel("Time [s]")
    #     ax[1,3].set_ylabel("Amplitude")

    #     ax[1,4].plot(t, h15, color='C0')
    #     ax[1,4].set_title("Estimation of recording")
    #     ax[1,4].set_xlabel("Time [s]")
    #     ax[1,4].set_ylabel("Amplitude")

    #     f = np.linspace(0, Fs_RX/1000, len(h11))
    #     ## first plot
    #     ax[2,0].plot(f, abs(H11), color='C0')
    #     ax[2,0].set_title("Frequency spectrum estimation")
    #     ax[2,0].set_xlabel("Frequency [Hz]")
    #     ax[2,0].set_ylabel("Amplitude")
    #     ax[2,0].set_ylim(bottom=0)

    #     ax[2,1].plot(f, abs(H12), color='C0')
    #     ax[2,1].set_title("Frequency spectrum estimation")
    #     ax[2,1].set_xlabel("Frequency [Hz]")
    #     ax[2,1].set_ylabel("Amplitude")
    #     ax[2,1].set_ylim(bottom=0)

    #     ax[2,2].plot(f, abs(H13), color='C0')
    #     ax[2,2].set_title("Frequency spectrum estimation")
    #     ax[2,2].set_xlabel("Frequency [Hz]")
    #     ax[2,2].set_ylabel("Amplitude")
    #     ax[2,2].set_ylim(bottom=0)

    #     ax[2,3].plot(f, abs(H14), color='C0')
    #     ax[2,3].set_title("Frequency spectrum estimation")
    #     ax[2,3].set_xlabel("Frequency [Hz]")
    #     ax[2,3].set_ylabel("Amplitude")
    #     ax[2,3].set_ylim(bottom=0)

    #     ax[2,4].plot(f, abs(H15), color='C0')
    #     ax[2,4].set_title("Frequency spectrum estimation")
    #     ax[2,4].set_xlabel("Frequency [Hz]")
    #     ax[2,4].set_ylabel("Amplitude")
    #     ax[2,4].set_ylim(bottom=0)

    #     plt.suptitle(title)
    #     fig.tight_layout()
    #     #plt.show()
    #     plt.savefig('plot_full_{}.png'.format(index), dpi=300)
    #     plt.close()




if __name__ == "__main__":
# Main block for testing
# Read the .wav file
# Localize the sound source
# Present the results
    localizer = localization()
    Fs, ABS1 = wavfile.read("opnames/record_x64_y40.wav")
    # ABS2 = wavaudioread("opnames/record_x82_y399.wav", Fs_RX)
    # ABS3 = wavaudioread("opnames/record_x109_y76.wav", Fs_RX)
    # ABS4 = wavaudioread("opnames/record_x143_y296.wav", Fs_RX)
    # ABS5 = wavaudioread("opnames/record_x150_y185.wav", Fs_RX)
    # ABS6 = wavaudioread("opnames/record_x178_y439.wav", Fs_RX)
    # ABS7 = wavaudioread("opnames/record_x232_y275.wav", Fs_RX)
    # ABS8 = wavaudioread("opnames/record_x4_y_hidden_1.wav", Fs_RX)
    # ABS9 = wavaudioread("opnames/record_x_y_hidden_2.wav", Fs_RX)
    # ABS10 = wavaudioread("opnames/record_x_y_hidden_3.wav", Fs_RX)
    #refsig = wavaudioread("opnames/reference.wav", Fs_RX)
    #FTrefsig = fft(refsig)

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



    #print(x_car1, y_car1)
    #print(x_car2, y_car2)
    #print(x_car3, y_car3)
    #print(x_car4, y_car4)
    #print(x_car5, y_car5)
    #print(x_car6, y_car6)
    #print(x_car7, y_car7)
    #print(x_car8, y_car8)
    #print(x_car9, y_car9)
    #print(x_car10, y_car10)

        