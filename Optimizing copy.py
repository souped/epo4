
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse, find_peaks       
from wavaudioread import wavaudioread
from recording_tool import recording_tool
from sympy import symbols, solve
import numpy as np
import math
import time
from scipy.signal import butter, filtfilt




class localization:
    def localization(audiowav, ref):
        # function goal: return the coorinates (x, y) using a ref signal and the audiosignal
                        
        """ plt.figure(figsize=(15, 5))
        for i in range(5):
            audio_channel_i = audiowav[:, i]
            segment = localization.detect_segments(audio_channel_i, 8)
            ch = localization.ch3(segment[3], ref)
            
            # Plot segment 5 in the first row
            plt.subplot(2, 5, i + 1)
            plt.plot(segment[3], label='Segment 5')
            plt.title(f'Channel {i+1} Segment 5')
            plt.legend()
            
            # Plot channel estimation in the second row
            plt.subplot(2, 5, i + 6)
            plt.plot(ch, label='Channel Estimation 5')
            plt.title(f'Channel {i+1} Estimation 5')
            plt.legend()

        plt.tight_layout()
        plt.show()"""


        """ plt.figure(figsize=(30, 10))

        for i in range(5):
            audio_channel_i = audiowav[:, i]
            segments = localization.detect_segments(audio_channel_i, 8)
            peaks_segments_i = localization.find_segment_peaks(segments) # get the peaks from the channel estimated segments

            for j in range(8):
                segment = segments[j]     
                peak_index_seg = peaks_segments_i[j]           

                plt.subplot(5, 8, i * 8 + j + 1)
                plt.plot(segment)
                plt.scatter(peak_index_seg, segment[peak_index_seg], color='red', zorder=5, label='Peak')
                plt.title(f'Channel {i+1} Segment {j+1}')
                plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(30, 10))

        for i in range(5):
            audio_channel_i = audiowav[:, i]
            segments = localization.detect_segments(audio_channel_i, 8)
            channel_responses_i = [localization.ch3(segment, ref) for segment in segments] # retrieve the channel estimation for each segment using the function ch3()
            peaks_ch_i = localization.find_segment_peaks(channel_responses_i)
            print(i, "ch:", peaks_ch_i)
            print(i, "seg:", peaks_segments_i)



            for j in range(8):
                peak_index = peaks_ch_i[j]
                ch = channel_responses_i[j]           

                plt.subplot(5, 8, i * 8 + j + 1)
                plt.plot(ch)
                plt.scatter(peak_index, ch[peak_index], color='red', zorder=5, label='Peak')
                plt.title(f'Channel {i+1} Segment {j+1}')
                plt.legend()
        plt.tight_layout()
        plt.show() """
        

        TDOA_list = []  
        
        #loop the calculateions so it will be done for a microphone and the following one, creating 10 microphone pairs (12, 13, 14, ... , 45)
        for i in range(5):
            for j in range(i+1, 5):
                # take one channel of the audio signal

                TDOA_list = localization.calculate_distances_for_channel_pairs(audiowav)


        # calculate the coordinates using the function coordinates_2d and the TDOA-list
        location = localization.coordinates_2d(TDOA_list)
        x_car = location[0]
        y_car = location[1]

        return x_car, y_car #return the coordinates
    def calculate_distances_for_channel_pairs(channels):
        cropped_channels = localization.crop_channels(channels)

        # calculate impulse response for each channel
        h = []
        for i in range(5):
            hi=abs(localization.ch3(ref, cropped_channels[i]))
            h.append(hi)

        TDOA = []
        for i in range(5):
            for j in range(i + 1, 5):  # Ensure pairs are unique and not repeated
                h0 = h[i]
                h1 = h[j]            
                dist = localization.calc_distance(h0, h1)
                TDOA.append((i+1, j+1, dist))
                #print(f"TDOA from microphone {i+1} to microphone {j+1}: {dist} meter")
        return TDOA

    def crop(recording):
        width = 10000
        recording = recording
        recording_tmp = recording[int(len(recording)/2):]
        peak = np.argmax(np.abs(recording_tmp)) + len(recording_tmp) 
        recording = recording[peak-width:peak+width]
        return recording
    
    def crop_channels(channels):
        width = 10000
        channels = [
            channels[0],
            channels[1],
            channels[2],
            channels[3],
            channels[4]
        ]
        
        base_channel = channels[0] # determine the same offset for all channels
        base_channel_tmp = base_channel[int(len(base_channel)/2):] # temporarily take the right part of the channel to avoid false peaks
        base_peak = np.argmax(np.abs(base_channel_tmp)) + len(base_channel_tmp) # determine the midpoint of all channels
        left_index = base_peak - width # offset from left
        right_index = base_peak + width # offset from right

        # fill cropped_channels with each individual channel of the corresponding recording
        cropped_channels = []
        for j in range(5):
            cropped_channels.append(channels[j][left_index:right_index]) # gets emptied after out of scope
        return cropped_channels

    def process_channel(channel_data, ref):
        # function goal: return the mean of the peaks (using segments) of the signal
        num_segments = 8
        segments = localization.detect_segments(channel_data, num_segments) # split signal into segments using function detect_segments()
        channel_responses = [localization.ch3(segment, ref) for segment in segments] # retrieve the channel estimation for each segment using the function ch3()
        channel_responses_array = np.array(channel_responses)
        peaks = localization.find_segment_peaks(channel_responses_array) # get the peaks from the channel estimated segments
        peaks.pop(np.argmax(peaks))
        peaks.pop(np.argmin(peaks))
        peaks.pop(np.argmax(peaks))
        peaks.pop(np.argmin(peaks))
        mean_peak = np.mean(peaks)
        return mean_peak
    

    def detect_segments(audio_signal, num_segments):
        # function goals: split the audio signal into segments
        """ start=0
        for i in range(len(audio_signal)):
            if np.abs(audio_signal[i])>0.2*np.max(audio_signal):
                start=i
                break
                
        audio_signal = audio_signal[start:start+num_segments*12000]
        print(start) """
        
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
        epsi = 0.15

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
        #h = h[0:Lhat]

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
    
    def coordinates_2d(tdoa,min_x=0,max_x=460,min_y=0,max_y=460,grid_resolution=5,finetuning=5):     
        #definition goal: return the coordinates of the car using the measured and calculated TDOA's  
        for i in range(finetuning):
            
            #set the grid dimensions using the given boundaries
            xgrid = np.tile(np.linspace(min_x,max_x,grid_resolution+2)[1:-1],grid_resolution)
            ygrid = np.repeat(np.linspace(min_y,max_y,grid_resolution+2)[1:-1],grid_resolution)
            zgrid = np.repeat(30,grid_resolution**2)
            grid_dimensions = np.stack((xgrid,ygrid,zgrid),axis=1)
            
            #manually calculate the TDOA's for each microphone-pair using the function gridTDOA
            gridTDOA = localization.TDOA_grid(grid_dimensions)
            
            #compare the calculated TDOA with the measure TDOA and find the point where their difference is the smallest
            comparison = np.linalg.norm(gridTDOA - tdoa, axis=1)
            best = grid_dimensions[np.argmin(comparison)]
            
            #To make the algorithm more accurate, once a point has been found, the algorithm will be looped
            #set the dimensions for a new grid to have a higher resolution around the found point (same gridpoints will be used for a smaller area)
            if i<finetuning:
                crop = 2/(grid_resolution**(i+1))
                min_x = best[0] - 460*crop
                max_x = best[0] + 460*crop
                min_y = best[1] - 460*crop
                max_y = best[1] + 460*crop
        return best




if __name__ == "__main__":  
# Main block for testing
# Read the .wav file
# Localize the sound source
# Present the results
    localizer = localization()


    start=time.time()
    Fref3, ref_signal3 = wavfile.read("gold_codes/gold_code_ref11.wav")
    ref_files = [
        #"gold_codes\\gold_code_ref2.wav",
        #"gold_codes\\gold_code_ref4.wav",
        #"gold_codes\\gold_code_ref6.wav",
        #"gold_codes\\gold_code_ref8.wav",
        #"gold_codes\\gold_code_ref10.wav",
        #"gold_codes\\gold_code_ref11.wav",
        #"gold_codes\\gold_code_ref12.wav",
        #"gold_codes\\gold_code_ref13.wav"
        #"extra test/12-06_ref2.wav",
        "extra test/ref1.wav"
        ]
    
    audio_files = [
        #"opnames nieuw\\gold_code13_test200-195.wav"
        # "opnames nieuw\\gold_code13_test128-375.wav",
        # "opnames nieuw\\gold_code13_test334-354.wav",
        #"failures\\failure1718378635.395296.wav",
        # "failures\\failure1718378639.9232068.wav",
        # "failures\\failure1718378644.446856.wav",
        #"failures\\failure1718378648.973517.wav",
        "extra test/12-06_x430_y317.wav",
        "extra test/12-06_x426_y36.wav",
        "extra test/12-06_x339_y157.wav",
        "extra test/12-06_x334_y354.wav",
        "extra test/12-06_x267_y258.wav",
        "extra test/12-06_x45_y267.wav"
    ]


    for ref in ref_files:
        Fs, audio = wavfile.read(ref)
        for i in range(len(audio)):
            if np.abs(audio[i])>50:
                start=i
                break

        pulse_length = 306

        print(f"{start}, {start+pulse_length}")
        
        segment = audio[48000:56000]
        #[start:start+pulse_length]

        # plt.plot(segment)
        # plt.xlabel('Sample Index')
        # plt.ylabel('Amplitude')
        # plt.title('Reference Signal')
        # plt.show()
        # plt.close()

        start = time.time()
        for file in audio_files:
            Fs, audio = wavfile.read(file)
            x_car, y_car = localization.localization(audio, segment)
            print(f"{file}, {ref}: x6 = {x_car}, y6 = {y_car}")
    end = time.time()
    print("Total time:", end - start)