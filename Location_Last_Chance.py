
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



class localization:
    @staticmethod 
    def localization(audiowav):
        # Split each recording into individual pulses
        num = 10
        cut = num*882000//40
        audio_channel1 = audiowav[cut:,0]
        audio_channel2 = audiowav[cut:,1]
        audio_channel3 = audiowav[cut:,2]
        audio_channel4 = audiowav[cut:,3]
        audio_channel5 = audiowav[cut:,4]
        Fref, ref_signal = wavfile.read("opnames/reference.wav")

        n=16
        ref =  ref_signal[755+22050*n:1470+22050*n,0]


        plt.figure(figsize=(10,5))
        plt.plot(audio_channel1, color='C0')
        plt.title("Input audio, with first 10 peaks deleted")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig("audio")
        plt.close()

        # plt.figure(figsize=(10,5))
        # plt.plot(audio1, color='C0')
        # plt.title("Input segment 5")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.savefig("seg 5")
        # plt.close()


        # plt.figure(figsize=(10,5))
        # plt.plot(ref, color='C0')
        # plt.title("Ref signal")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.show()

        plt.figure(figsize=(10,5))
        plt.plot(ref, color='C0')
        plt.title("Ref signal small part")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig("Ref")
        plt.close()

        
        #channel estimation
        eps = 0.01
        channel_responses_1 = localization.ch3(audio_channel1, ref, eps)
        channel_responses_2 = localization.ch3(audio_channel2, ref, eps)
        channel_responses_3 = localization.ch3(audio_channel3, ref, eps)
        channel_responses_4 = localization.ch3(audio_channel4, ref, eps)
        channel_responses_5 = localization.ch3(audio_channel5, ref, eps)
        

        #peaks
        max_peaks = 40-num
        peaks_channel1 = localization.find_peaks(channel_responses_1, max_peaks)
        peaks_channel2 = localization.find_peaks(channel_responses_2, max_peaks)
        peaks_channel3 = localization.find_peaks(channel_responses_3, max_peaks)
        peaks_channel4 = localization.find_peaks(channel_responses_4, max_peaks)
        peaks_channel5 = localization.find_peaks(channel_responses_5, max_peaks)

        # # Plotting
        # epsi_values = np.linspace(0.001, 0.01, num=6)
        # cmap = plt.get_cmap('rainbow')
        # plt.figure(figsize=(20, 6))
        # plt.plot(audio_channel1)

        # for i, epsi in enumerate(epsi_values):
        #     h = localization.ch3(channel_responses_1, ref, epsi)
        #     peaks_indices = np.where(h == np.max(h))[0]  # Find all peak indices
        #     for peak in peaks_indices:
        #         color = cmap(i / len(epsi_values))
        #         plt.axvline(x=peak, color=color, linestyle='--', alpha=0.5, label=f"Epsi={epsi}")  # Plot vertical line at each peak index
        #         #legend_handles.append(plt.axvline(x=peak, color='r', linestyle='--', alpha=0.5))
            

        # plt.xlabel('Index')
        # plt.ylabel('Magnitude')
        # plt.title('Channel Estimation for Different Epsi Values')
        # plt.legend()
        # plt.grid(True)
        # #plt.legend(handles=legend_handles, loc='upper right')
        # plt.show()

        plt.figure(figsize=(10,5))
        # plt.plot((audio_channel1/60000)[202000:206000], color='C1', label='Audio channel, normalized and divided by 100')
        
        plt.plot(channel_responses_1, color='C0', label ='Channel estimation')
        plt.title("Channel estimation")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        for peak in peaks_channel1:
            plt.axvline(x=peak, color='r', linestyle='-.', label='Vertical Line')
        plt.grid(True)
        # plt.legend()
        plt.savefig("ch1_5")
        plt.close()

        # plt.figure(figsize=(10,5))
        # plt.plot(channel_responses_1, color='C0')
        # for peak in peaks_channel1:
        #     plt.axvline(x=peak, color='r', linestyle='--', label='')
        # plt.title("Channel Response - Channel 1")
        # plt.xlabel("Sample Index")
        # plt.ylabel("Amplitude")
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        # Calculate TDOA between different microphone pairs
        TDOA12 = localization.TDOA(peaks_channel1, peaks_channel2)
        TDOA13 = localization.TDOA(peaks_channel1, peaks_channel3)
        TDOA14 = localization.TDOA(peaks_channel1, peaks_channel4)
        # TDOA23 = localization.TDOA(mean_peak_2, mean_peak_3)
        # TDOA24 = localization.TDOA(mean_peak_2, mean_peak_4)
        # TDOA34 = localization.TDOA(mean_peak_3, mean_peak_4)
        
        # print("D12 = ", TDOA12)
        # print("D13 = ", TDOA13)
        # print("D14 = ", TDOA14)
        
        x, y = localization.coordinate_2d(TDOA12, TDOA13, TDOA14)
        
        return x, y
    
    @staticmethod 
    def detect_segments(audio_signal):
        # remainder = len(audio_signal) % 40
        # if remainder != 0:
        #     audio_signal = audio_signal[:-remainder]
        segments = np.array((np.array_split(np.abs(audio_signal), 20)))
        # trimmed_segments = [segment[4000:15000] for segment in segments]

        # Converteer de gesnoeide segmenten naar een numpy array
        # trimmed_segments_array = np.array(trimmed_segments)
    

        return segments

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
        print(peaks)
        return peaks

    @staticmethod 
    def TDOA(peak1, peak2):
        tdoas = [(peak_2 - peak_1) / Fs for peak_1, peak_2 in zip(peak1, peak2)]
        # Bereken het gemiddelde van de TDOA's
        mean_tdoa = np.mean(tdoas)
        D = mean_tdoa *343
        return D
       
                
    @staticmethod 
    def ch3(y, x, epsi):
        Nx = len(x)           # Length of x
        Ny = len(y)           # Length of y
        L = Ny - Nx + 1          # Length of h
        #Lhat = max(len(y), len(x)) 


        # Force x to be the same length as y
        x = np.append(x, [0]*(L-1))   # Make x same length as y
        #print(len(reference_signal))



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

    @staticmethod 
    def coordinate_2d(D12, D13, D14):
        
        # When D12 is equal to 0, the y coordinate is 2.4, to calculate x we used D14 = sqrt((4.8-x)^2+2.4^2)-sqrt(x^2+2.4^)
        if D12 == 0:
            if D14 > 0:
                x = (-13824 + 600*D14**2 + 5*(np.sqrt(624*D14**6 - 43200*D14**4+663552*D14**2)))/(10*(25*D14**2-576))
                y = 2.4
            elif D14 < 0:
                x = (-13824 + 600*D14**2 - 5*(np.sqrt(624*D14**6 - 43200*D14**4+663552*D14**2)))/(10*(25*D14**2-576))
                y = 2.4
            else:
                x = 2.4
                y = 2.4
        # The same goes for D14, to calculate the x coordinate
        elif D14 == 0:
            if D12 > 0:
                x = 2.4
                y = (-13824 + 600*D12**2 + 5*(np.sqrt(624*D12**6 - 43200*D12**4+663552*D12**2)))/(10*(25*D12**2-576))
            elif D12 < 0:
                x = 2.4
                y = (-13824 + 600*D12**2 - 5*(np.sqrt(624*D12**6 - 43200*D12**4+663552*D12**2)))/(10*(25*D12**2-576))
            else:
                x = 2.4
                y = 2.4   
        
        else:
            D23 = D13 - D12
            D24 = D14 - D12
            D34 = D14 - D13

            # Locations of microphones
            X1 = np.array([0, 0])
            X2 = np.array([0, 4.8])
            X3 = np.array([4.8, 4.8])
            X4 = np.array([4.8, 0])
            
            # Calculate 2D coordinates based on TDOA measurements
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
            
            X21 = X2 - X1
            X31 = X3 - X1
            X41 = X4 - X1
            X32 = X3 - X2
            X42 = X4 - X2
            X43 = X4 - X3
            
            A = np.array([
                [2 * X21[0], 2 * X21[1], -2 * D12, 0, 0],
                [2 * X31[0], 2 * X31[1], 0, -2 * D13, 0],
                [2 * X41[0], 2 * X41[1], 0, 0, -2 * D14],
                [2 * X32[0], 2 * X32[1], 0, -2 * D23, 0],
                [2 * X42[0], 2 * X42[1], 0, 0, -2 * D24],
                [2 * X43[0], 2 * X43[1], 0, 0, -2 * D34]
            ])
            A_inv = np.linalg.pinv(A)
            result = np.dot(A_inv, B)
            x = result[0, 0]
            y = result[1, 0]

        return x, y
        


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

    x_car1, y_car1 = localization.localization(ABS1)
    print("Coordinates_x64_y40 : x = ", x_car1, ", y = ", y_car1)
    # x_car2, y_car2 = localization.localization(ABS2)
    # print("Coordinates_x82_y399 : x = ", x_car2, ", y = ", y_car2)
    # x_car3, y_car3 = localization.localization(ABS3)
    # print("Coordinates_x109_y76 : x = ", x_car3, ", y = ", y_car3)
    # x_car4, y_car4 = localization.localization(ABS4)
    # print("Coordinates_x143_y296 : x = ", x_car4, ", y = ", y_car4)
    # x_car5, y_car5 = localization.localization(ABS5)
    # print("Coordinates_x150_y185 : x = ", x_car5, ", y = ", y_car5)
    # x_car6, y_car6 = localization.localization(ABS6)
    # print("Coordinates_x178_y439 : x = ", x_car6, ", y = ", y_car6)
    # x_car7, y_car7 = localization.localization(ABS7)
    # print("Coordinates_x232_y275 : x = ", x_car7, ", y = ", y_car7)
    # x_car8, y_car8 = localization.localization(ABS8)
    # print("Coordinates_x4_y_hidden_1 : x = ", x_car8, ", y = ", y_car8)
    # x_car9, y_car9 = localization.localization(ABS9)
    # print("Coordinates_x_y_hidden_2 : x = ", x_car9, ", y = ", y_car9)
    # x_car10, y_car10 = localization.localization(ABS10)
    # print("Coordinates_x_y_hidden_3 : x = ", x_car10, ", y = ", y_car10)

   
        