import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft,ifft
from scipy.signal import convolve,unit_impulse,find_peaks
# from IPython.display import Audio
from refsignal import refsignal  # model for the EPO4 audio beacon signal
from wavaudioread import wavaudioread
# from recording_tool import recording_tool
# from sympy import symbols,solve
import numpy as np
import math
import time

Fs = 48000

class localization:
    # def __init__(recording, debug=False):
    def localization(self,audiowav,ref):
        TDOA_list=[]
        for i in range(5):
            for j in range(i+1, 5):
                # Split each recording into individual pulses
                audio_channel_i = audiowav[:, i]
                audio_channel_j = audiowav[:, j]
                # calculate the mean of the list with peaks using the function process_channel() for each channel
                mean_peak_i = localization.process_channel(audio_channel_i, ref)
                mean_peak_j = localization.process_channel(audio_channel_j, ref)
                # using the peaks, calculate the TDOA's by comparing the microphone pairs using function TDOA()
                TDOA = localization.TDOA(mean_peak_j, mean_peak_i)
                TDOA_list.append(TDOA)
        # calculate the coordinates using the function coordinates_2d and the TDOA-list
        location = localization.coordinates_2d(TDOA_list)
        x_car = location[0]
        y_car = location[1]
        return x_car,y_car

    def process_channel(channel_data,ref):
        # function goal: return the mean of the peaks (using segments) of the signal
        num_segments = 8
        segments = localization.detect_segments(channel_data, num_segments) # split signal into segments using function detect_segments()
        channel_responses = [localization.ch3(segment, ref) for segment in segments] # retrieve the channel estimation for each segment using the function ch3()
        channel_responses_array = np.array(channel_responses)
        peaks = localization.find_segment_peaks(channel_responses_array) # get the peaks from the channel estimated segments
        trimmed_peaks = peaks.pop(np.argmax(peaks))
        trimmed_peaks = peaks.pop(np.argmin(peaks))
        mean_peak = np.mean(trimmed_peaks)
        return mean_peak

    def detect_segments(audio_signal, num_segments):
        # function goals: split the audio signal into segments
        segments = []
        segment_length = len(audio_signal) // num_segments
        segments = [(audio_signal[i*segment_length : (i+1)*segment_length]) for i in range(num_segments)]
        return segments

    def find_segment_peaks(segment_signal):
        # function goal: return a list of indeces of the maximums of each segment
        peaks_list = []
        for segment in segment_signal:
            peaks = np.argmax(segment)
            peaks_list.append(peaks)
        return peaks_list

    def TDOA(peak1, peak2):
        # function goal: return the TDOA using 2 peaks and the sampling freuency
        return (peak2 - peak1)/Fs

    @staticmethod
    def ch3(signal_1, reference_signal):
        Nsignal_1 = len(signal_1)
        Nreference_signal = len(reference_signal)
        L = Nsignal_1 - Nreference_signal + 1
        Lhat = max(len(reference_signal), len(signal_1)) 
        epsi = 0.005
        # Force x to be the same length as y
        reference_signal = np.append(reference_signal, [0]* (L-1))
        # Deconvolution in frequency domain
        fft_signal_1 = fft(signal_1)
        fft_reference_signal = fft(reference_signal)
        # Threshold to avoid blow ups of noise during inversion
        ii = (np.abs(fft_reference_signal)) < (np.max(np.abs(fft_reference_signal))*epsi)
        H = np.divide(fft_signal_1, fft_reference_signal)
        H[ii] = 0
        h = np.real(ifft(H))    
        h = h[0:Lhat]
        return h


    def TDOA_grid(grid_dimensions):
        #definition goal: return the manually calculated TDOA values for the given grid
        gridTDOA = []
        mic_locs = np.array([[0,0,50],[0,460,50],[460,460,50],[460,0,50],[0,230,80]]) #set mic locations
        # take the distance (norm) between the microphones and the grid-locations
        for row in grid_dimensions:
            distances = []
            for loc in mic_locs:
                dist = np.linalg.norm(loc-row)
                distances.append(dist)
            #conversion of distance (cm) to time (s) using speed of sound (m/s)
            times =  np.array(distances)/34300
            #calculate the TDOA (time differences) between each microphone pair
            TDOA = []
            for i in range(0,len(times)):
                for j in range(i+1,len(times)):
                    TDOA = np.append(TDOA,(times[i]-times[j]))
            gridTDOA = np.concatenate((gridTDOA,TDOA))
        #reshape the list to a matrix so each column is corresponding to a different microphone pair
        gridTDOA = np.reshape(gridTDOA,(-1,10))
        return(gridTDOA)

    def coordinates_2d(tdoa,size=10,min_x=0,max_x=460,min_y=0,max_y=460,finetuning=4):
        for i in range(5):
            xgrid = np.tile(np.linspace(min_x,max_x,size+2)[1:-1],size)
            ygrid = np.repeat(np.linspace(min_y,max_y,size+2)[1:-1],size)
            zgrid = np.repeat(30,size**2)
            grid_dimensions = np.stack((xgrid,ygrid,zgrid),axis=1)
            #manually calculate the TDOA's for each microphone-pair using the function gridTDOA
            gridTDOA = localization.TDOA_grid(grid_dimensions)
            #compare the calculated TDOA with the measure TDOA and find the point where their difference is the smallest
            errors = np.linalg.norm(gridTDOA - tdoa, axis=1)
            best = grid_dimensions[np.argmin(errors)]
            #To make the algorithm more accurate, once a point has been found, the algorithm will be looped
            #set the dimensions for a new grid to have a higher resolution around the found point (same gridpoints will be used for a smaller area)
            if i<finetuning:
                padding = 460/(size**(i+1))/2
                min_x = best[0] - padding
                max_x = best[0] + padding
                min_y = best[1] - padding
                max_y = best[1] + padding
        return best


if __name__=="__main__":
    # Main block for testing
    # Read the .wav file
    # Localize the sound source
    # Present the results
    localizer=localization()

    start=time.time()
    Fref,ref_signal=wavfile.read("gold_codes\gold_code_ref13.wav")
    ref_signal=ref_signal[:,1]
    ref=ref_signal[8600:9000]
    print(type(ref))
    audio_files=[
        "gold_code13_test128-375.wav",
    ]

    start=time.time()
    for file in audio_files:
        Fs,audio=wavfile.read(file)
        print(type(audio))
        plt.plot(audio)
        plt.title(f"Reference signal for {file}")
        plot_filename=file.replace("Beacon/","").replace(".wav",".png")
        plt.savefig(plot_filename)
        plt.close()

        x_car,y_car=localization.localization(audio,ref)
        print(f"{file}: x = {x_car}, y = {y_car}")

    end=time.time()
    print("Total time:",end - start)