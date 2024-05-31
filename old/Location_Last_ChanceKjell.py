
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse, find_peaks, correlate
#from IPython.display import Audio
from refsignal import refsignal            # model for the EPO4 audio beacon signal
from wavaudioread import wavaudioread
from recording_tool import recording_tool
from sympy import symbols, solve
import numpy as np
import math
import time



class localization:
    @staticmethod 
    def localization(audiowav):
        # Split each recording into individual pulses
        num = 10
        cut = num*882000//40
        Fref, ref_signal = wavfile.read("opnames/reference.wav")

        n=16
        ref =  ref_signal[755+22050*n:1470+22050*n,0]
        
        #channel estimation
        eps = 0.01
        

        #peaks
        max_peaks = 40-num


        TDOA_list = []
        for i in range(5):
            for j in range(i+1, 5):
                audio_channel_i = audiowav[cut:,i]
                channel_responses_i = localization.ch3(audio_channel_i, ref, eps)
                peaks_channel_i = localization.find_peaks(channel_responses_i, max_peaks)
                mean_peak_i = np.mean(peaks_channel_i)

                audio_channel_j = audiowav[cut:,j]
                channel_responses_j = localization.ch3(audio_channel_j, ref, eps)
                peaks_channel_j = localization.find_peaks(channel_responses_j, max_peaks)
                mean_peak_j = np.mean(peaks_channel_j)

                TDOA = localization.TDOA(mean_peak_j, mean_peak_i)
                TDOA_list.append(TDOA)
        x, y = localization.coordinates_2d(TDOA_list)
        
        return x, y

    def TDOA_grid(grid_dimensions):
        gridTDOA = []
        mic_locs = np.array([[0,0,50],[0,480,50],[480,480,50],[480,0,50],[0,240,80]])
        for row in grid_dimensions:
            distances = []
            for loc in mic_locs:
                dist = np.linalg.norm(loc-row)
                distances.append(dist)
            times =  np.array(distances)/34300
            TDOA = []
            for i in range(0,len(times)):
                for j in range(i+1,len(times)):
                    TDOA = np.append(TDOA,(times[i]-times[j]))
            gridTDOA = np.concatenate((gridTDOA,TDOA))
        gridTDOA = np.reshape(gridTDOA,(-1,10))
        return(gridTDOA)
    
    def coordinates_2d(tdoa,size=10,min_x=0,max_x=480,min_y=0,max_y=480,finetuning=5): 
        for i in range(5):
            xgrid = np.tile(np.linspace(min_x,max_x,size+2)[1:-1],size)
            ygrid = np.repeat(np.linspace(min_y,max_y,size+2)[1:-1],size)
            zgrid = np.repeat(30,size**2)
            grid_dimensions = np.stack((xgrid,ygrid,zgrid),axis=1)

            gridTDOA = localization.TDOA_grid(grid_dimensions)

            error_list = np.array([])
            for row in gridTDOA:
                error = np.linalg.norm(row-tdoa)
                error_list = np.append(error_list,error)
            best = grid_dimensions[np.argmin(error_list)]

            if i<finetuning:
                padding = 480/(size**(i+1))/2
                min_x = best[0] - padding
                max_x = best[0] + padding
                min_y = best[1] - padding
                max_y = best[1] + padding

        x=best[0]
        y=best[1]
        return x, y



    @staticmethod 
    def find_segment_peaks(segment_signal):
        peaks_list = []
        for segment in segment_signal:
            max_value = np.max(segment)
            # Bereken de drempelwaarde (70% van de maximumwaarde)
            threshold_value = 0.7 * max_value
    
            # Vind de eerste index waar de waarde van het segment de drempel overschrijdt
            peak_index = np.argmax(segment >= threshold_value)
            print(peak_index)
            peaks_list.append(peak_index)
              
        return peaks_list
    
    @staticmethod 
    def find_peaks(signal, max_peaks, threshold_ratio=0.8, skip_indices=22000):
        peaks = []
        current_index = 0
        signal_length = len(signal)
        
        while current_index < signal_length and len(peaks) < max_peaks:
            # Bereken de maximumwaarde van het resterende signaal
            segment = signal[current_index:current_index + skip_indices]
            max_value = np.max(segment)
            
            
            # Bereken de drempelwaarde (70% van de maximumwaarde)
            threshold_value = threshold_ratio * max_value
            
            # Vind de eerste index waar de waarde van het signaal de drempel overschrijdt
            for i in range(current_index, signal_length):
                if signal[i] >= threshold_value:
                    peaks.append(i)
                    current_index = i + skip_indices  # Skip de volgende 22000 indexen
                    break
            else:
                    # Geen piek gevonden die aan de drempel voldoet
                break
        return peaks

    @staticmethod 
    def TDOA(peak1, peak2):
        mean_tdoa = (peak2 - peak1)/Fs
        return mean_tdoa
       
                
    @staticmethod 
    def ch3(y, x, epsi):
        Nx = len(x)           # Length of x
        Ny = len(y)           # Length of y
        L = Ny - Nx + 1          # Length of h
        #Lhat = max(len(y), len(x)) 


        # Force x to be the same length as y
        x = np.append(x, [0]*(L-1))   # Make x same length as y

        # Deconvolution in frequency domain
        X = fft(x)
        Y = fft(y)

        # Threshold to avoid blow ups of noise during inversion
        ii = np.absolute(X) < epsi*max(np.absolute(X))

        H = Y/X
        H[ii] = 0

        #H = [0 if condition else reference_signal for reference_signal, condition in zip(fft_signal_1, ii)]
        h = np.real(ifft(H))    
        #h = h[0:Lhat]

        return h

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


    start = time.time()
    x_car1, y_car1 = localization.localization(ABS1)
    print("Coordinates_x64_y40 : x = ", x_car1, ", y = ", y_car1)
    x_car2, y_car2 = localization.localization(ABS2)
    print("Coordinates_x82_y399 : x = ", x_car2, ", y = ", y_car2)
    x_car3, y_car3 = localization.localization(ABS3)
    print("Coordinates_x109_y76 : x = ", x_car3, ", y = ", y_car3)
    x_car4, y_car4 = localization.localization(ABS4)
    print("Coordinates_x143_y296 : x = ", x_car4, ", y = ", y_car4)
    x_car5, y_car5 = localization.localization(ABS5)
    print("Coordinates_x150_y185 : x = ", x_car5, ", y = ", y_car5)
    x_car6, y_car6 = localization.localization(ABS6)
    print("Coordinates_x178_y439 : x = ", x_car6, ", y = ", y_car6)
    x_car7, y_car7 = localization.localization(ABS7)
    print("Coordinates_x232_y275 : x = ", x_car7, ", y = ", y_car7)
    x_car8, y_car8 = localization.localization(ABS8)
    print("Coordinates_x4_y_hidden_1 : x = ", x_car8, ", y = ", y_car8)
    x_car9, y_car9 = localization.localization(ABS9)
    print("Coordinates_x_y_hidden_2 : x = ", x_car9, ", y = ", y_car9)
    x_car10, y_car10 = localization.localization(ABS10)
    print("Coordinates_x_y_hidden_3 : x = ", x_car10, ", y = ", y_car10)
    end= time.time()
    print("time= ", end-start)

   
        