
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
        Fref, ref_signal = wavfile.read("opnames/reference.wav")
        ref_signal =  ref_signal[:,0]
        refsig = localization.detect_segments(ref_signal)
        ref = refsig[12]
        ref = ref[750:1500]


    
        TDOA_list = []
        # Calculate TDOA between different microphone pairs
        for i in range(5):
            for j in range(i+1, 5):
                audio_channel_i = audiowav[:,i]
                segments_channel_i = localization.detect_segments(audio_channel_i)
                segments_channel_i = segments_channel_i[5:35]
                channel_responses_i = [localization.ch3(segment, ref) for segment in segments_channel_i]
                channel_responses_array_i = np.array(channel_responses_i)
                peaks_channel_i = localization.find_segment_peaks(channel_responses_array_i)
                sorted_peaks_i = np.sort(peaks_channel_i)
                trimmed_peaks_i = sorted_peaks_i[10:-10]
                mean_peak_i = np.mean(trimmed_peaks_i)

                audio_channel_j = audiowav[:,j]
                segments_channel_j = localization.detect_segments(audio_channel_j)
                segments_channel_j = segments_channel_j[5:35]
                channel_responses_j = [localization.ch3(segment, ref) for segment in segments_channel_j]
                channel_responses_array_j = np.array(channel_responses_j)
                peaks_channel_j = localization.find_segment_peaks(channel_responses_array_j)
                sorted_peaks_j = np.sort(peaks_channel_j)
                trimmed_peaks_j = sorted_peaks_j[10:-10]
                mean_peak_j = np.mean(trimmed_peaks_j)

                TDOA = localization.TDOA(mean_peak_j, mean_peak_i)
                TDOA_list.append(TDOA)

        location = localization.coordinates_2d(TDOA_list)

        return location
    
    def detect_segments(audio_signal):
        segments = []
        num_segments = 40
        segment_length = len(audio_signal) // num_segments
        segments = [abs(audio_signal[i*segment_length : (i+1)*segment_length]) for i in range(num_segments)]
        return segments

    def find_segment_peaks(segment_signal):
        peaks_list = []
        for segment in segment_signal:
            peaks = np.argmax(segment)
            peaks_list.append(peaks)
              
        return peaks_list
    
    def TDOA(peak1, peak2):
        mean_tdoa = (peak2 - peak1)/Fs
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
        #print(len(reference_signal))
        
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


        #old way of coordinate derivation (Matrix), turned out to be too inaccurate
    """ def coordinate_2d(D12, D13, D14):
        D23= D13-D12
        D24= D14-D12
        D34= D14-D13

        # Calculate 2D coordinates based on TDOA measurements
        X1 = np.array([0, 0])
        X2 = np.array([0, 4.8])
        X3 = np.array([4.8, 4.8])
        X4 = np.array([4.8, 0])
        #X5 = np.array([0, 2.4])
        
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

        A_inv = np.linalg.pinv(A)
        result = np.dot(A_inv, B)
        x = result[0,0]
        y = result[1,0]
        
        # print(A)
        # print(A_inv)
        
        return x, y """
    
    def TDOA_calc(xyz):
        allTDOA = []
        mic_locs = np.array([[0,0,50],[0,480,50],[480,480,50],[480,0,50],[0,240,80]])
        for row in xyz:
            distances = []
            for loc in mic_locs:
                dist = np.linalg.norm(loc-row)
                distances.append(dist)
            times =  np.array(distances)/34120
            TDOA = []
            for i in range(0,len(times)):
                for j in range(i+1,len(times)):
                    TDOA = np.append(TDOA,(times[i]-times[j]))
            allTDOA = np.concatenate((allTDOA,TDOA))
        allTDOA = np.reshape(allTDOA,(-1,10))
        return(allTDOA)
    
    def coordinates_2d(tdoa,size=10,min_x=0,max_x=480,min_y=0,max_y=480,iteration=0,cont_count=4): 
        xgrid = np.tile(np.linspace(min_x,max_x,size+2)[1:-1],size)
        ygrid = np.repeat(np.linspace(min_y,max_y,size+2)[1:-1],size)
        zgrid = np.repeat(30,size**2)
        grid = np.stack((xgrid,ygrid,zgrid),axis=1)

        gridTDOA = localization.TDOA_calc(grid)

        errors = np.array([])
        for row in gridTDOA:
            error = np.linalg.norm(row-tdoa)
            errors = np.append(errors,error)
        best = grid[np.argmin(errors)]

        if iteration<cont_count:
            padding = 480/(size**(iteration+1))/2
            min_x = best[0] - padding
            max_x = best[0] + padding
            min_y = best[1] - padding
            max_y = best[1] + padding
            
            iteration += 1
            best = localization.coordinates_2d(tdoa,size,min_x,max_x,min_y,max_y,iteration,cont_count)
            return(best)
        else: return(best)


if __name__ == "__main__":
# Main block for testing
# Read the .wav file
# Localize the sound source
# Present the results
    localizer = localization()
    Fs, ABS1 = wavfile.read("opnames/record_x64_y40.wav")
    Fs, ABS2 = wavfile.read("opnames/record_x82_y399.wav")
    Fs, ABS3 = wavfile.read("opnames/record_x109_y76.wav")
    Fs, ABS4 = wavfile.read("opnames/record_x143_y296.wav")
    Fs, ABS5 = wavfile.read("opnames/record_x150_y185.wav")
    Fs, ABS6 = wavfile.read("opnames/record_x178_y439.wav")
    Fs, ABS7 = wavfile.read("opnames/record_x232_y275.wav")
    Fs, ABS8 = wavfile.read("opnames/record_x4_y_hidden_1.wav")
    Fs, ABS9 = wavfile.read("opnames/record_x_y_hidden_2.wav")
    Fs, ABS10 = wavfile.read("opnames/record_x_y_hidden_3.wav")


    x_car1, y_car1, z = localization.localization(ABS1)
    print("Coordinates_x64_y40 : x = ", x_car1, ", y = ", y_car1)
    x_car2, y_car2, z = localization.localization(ABS2)
    print("Coordinates_x82_y399 : x = ", x_car2, ", y = ", y_car2)
    x_car3, y_car3, z = localization.localization(ABS3)
    print("Coordinates_x109_y76 : x = ", x_car3, ", y = ", y_car3)
    x_car4, y_car4, z = localization.localization(ABS4)
    print("Coordinates_x143_y296 : x = ", x_car4, ", y = ", y_car4)
    x_car5, y_car5, z = localization.localization(ABS5)
    print("Coordinates_x150_y185 : x = ", x_car5, ", y = ", y_car5)
    x_car6, y_car6, z = localization.localization(ABS6)
    print("Coordinates_x178_y439 : x = ", x_car6, ", y = ", y_car6)
    x_car7, y_car7, z = localization.localization(ABS7)
    print("Coordinates_x232_y275 : x = ", x_car7, ", y = ", y_car7)
    x_car8, y_car8, z = localization.localization(ABS8)
    print("Coordinates_x4_y_hidden_1 : x = ", x_car8, ", y = ", y_car8)
    x_car9, y_car9, z = localization.localization(ABS9)
    print("Coordinates_x_y_hidden_2 : x = ", x_car9, ", y = ", y_car9)
    x_car10, y_car10, z = localization.localization(ABS10)
    print("Coordinates_x_y_hidden_3 : x = ", x_car10, ", y = ", y_car10)
