Here are the function descriptions with `"""..."""` comments for each function in your `localization` class:

```python
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

class localization:
    def localization(audiowav, reference):
        """
        Function goal: Localize the sound source and return its coordinates (x, y).

        Function inputs:
        - audiowav: 2D array, audio data from multiple microphones (5-channels)
        - reference: 1D array, reference signal

        Function outputs:
        - x_car: float, estimated x-coordinate
        - y_car: float, estimated y-coordinate
        """
        TDOA_list = []  
        
        # Loop to calculate TDOA for microphone pairs (12, 13, 14, ..., 45)
        for i in range(5):
            for j in range(i+1, 5):
                # Take one channel of the audio signal
                audio_channel_i = audiowav[:, i]
                audio_channel_j = audiowav[:, j]

                # Calculate the mean of the peaks using the function process_channel() for each channel
                mean_peak_i = localization.process_channel(audio_channel_i, reference)
                mean_peak_j = localization.process_channel(audio_channel_j, reference)

                # Calculate TDOA by comparing microphone pairs using function TDOA()
                TDOA = localization.TDOA(mean_peak_j, mean_peak_i)
                TDOA_list.append(TDOA)

        # Calculate the coordinates using function coordinates_2d and the TDOA-list
        location = localization.coordinates_2d(TDOA_list)
        x_car = location[0]
        y_car = location[1]

        return x_car, y_car # Return the coordinates
    
    def process_channel(channel_data, ref):
        """
        Function goal: Process a channel of audio data to estimate the mean of the peaks.

        Function inputs:
        - channel_data: 1D array, audio data from a single microphone channel.
        - ref: 1D array, reference signal

        Function outputs:
        - mean_peak: float, mean value of the detected peaks in the channel data.
        """
        num_segments = 8
        segments = localization.detect_segments(channel_data, num_segments) # Split signal into segments using function detect_segments()
        channel_responses = [localization.ch3(segment, ref) for segment in segments] # Retrieve the channel estimation for each segment using function ch3()
        channel_responses_array = np.array(channel_responses)
        peaks = localization.find_segment_peaks(channel_responses_array) # Get the peaks from the channel estimated segments
        peaks.pop(np.argmax(peaks))
        peaks.pop(np.argmin(peaks))
        peaks.pop(np.argmax(peaks))
        peaks.pop(np.argmin(peaks))
        mean_peak = np.mean(peaks)
        return mean_peak
    
    def detect_segments(audio_signal, num_segments):
        """
        Function goal: Split an audio signal into segments.

        Function inputs:
        - audio_signal: 1D array, audio data from a single microphone channel.
        - num_segments: int, number of segments to divide the audio signal into.

        Function outputs:
        - segments: list of 1D arrays, segmented audio data.
        """
        segments = []
        segment_length = len(audio_signal) // num_segments
        segments = [(audio_signal[i*segment_length : (i+1)*segment_length]) for i in range(num_segments)]
        return segments

    def find_segment_peaks(segment_signal):
        """
        Function goal: Find peaks in each segment of a signal.

        Function inputs:
        - segment_signal: 2D array, signal segments to analyze.

        Function outputs:
        - peaks_list: list of ints, indices of the maximums of each segment.
        """
        peaks_list = []
        for segment in segment_signal:
            peaks = np.argmax(segment)
            peaks_list.append(peaks)
        return peaks_list
    
    def TDOA(peak1, peak2):
        """
        Function goal: Calculate Time Difference of Arrival (TDOA) using peak indices.

        Function inputs:
        - peak1: int, index of the first peak.
        - peak2: int, index of the second peak.

        Function outputs:
        - tdoa: float, calculated Time Difference of Arrival.
        """
        return (peak2 - peak1)/Fs
            
    @staticmethod 
    def ch3(signal_1, reference_signal):
        """
        Function goal: Perform channel estimation using deconvolution in the frequency domain.

        Function inputs:
        - signal_1: 1D array, signal from a microphone channel.
        - reference_signal: 1D array, reference signal used for deconvolution.

        Function outputs:
        - h: 1D array, estimated channel response.
        """
        Nsignal_1 = len(signal_1)
        Nreference_signal = len(reference_signal)

        L = Nsignal_1 - Nreference_signal + 1
        Lhat = max(len(reference_signal), len(signal_1)) 
        epsi = 0.05

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
        """
        Function goal: Calculate TDOA values for a grid of locations.

        Function inputs:
        - grid_dimensions: 2D array, grid dimension limits

        Function outputs:
        - gridTDOA: 2D array, TDOA values for each microphone pair corresponding to the grid.
        """
        gridTDOA = []
        mic_locs = np.array([[0,0,50],[0,460,50],[460,460,50],[460,0,50],[0,230,80]]) # Set mic locations
        
        # Calculate distance (norm) between microphones and grid locations
        for row in grid_dimensions:
            distances = []
            for loc in mic_locs:
                dist = np.linalg.norm(loc-row)
                distances.append(dist)
            # Conversion of distance (cm) to time (s) using speed of sound (m/s)
            times =  np.array(distances)/34300
            
            # Calculate TDOA (time differences) between each microphone pair
            TDOA = []
            for i in range(0,len(times)):
                for j in range(i+1,len(times)):
                    TDOA = np.append(TDOA,(times[i]-times[j]))
            gridTDOA = np.concatenate((gridTDOA,TDOA))
        
        # Reshape the list to a matrix so each column corresponds to a different microphone pair
        gridTDOA = np.reshape(gridTDOA,(-1,10))
        return(gridTDOA)
    
    def coordinates_2d(tdoa,min_x=0,max_x=460,min_y=0,max_y=460,grid_resolution=5,finetuning=5):     
        """
        Function goal: Calculate the 2D coordinates of the sound source using measured and calculated TDOAs.

        Function inputs:
        - tdoa: 1D array, measured TDOA values for microphone pairs.
        - min_x, max_x: float, boundaries for x-coordinate.
        - min_y, max_y: float, boundaries for y-coordinate.
        - grid_resolution: int, number of grid points
        - finetuning: int, number of iterations for fine-tuning the coordinate estimate.

        Function outputs:
        - best: 1D array, best estimated coordinates (x, y) of the sound source.
        """
        for i in range(finetuning):
            # Set the grid dimensions using the given boundaries
            xgrid = np.tile(np.linspace(min_x,max_x,grid_resolution+2)[1:-1],grid_resolution)
            ygrid = np.repeat(np.linspace(min_y,max_y,grid_resolution+2)[1:-1],grid_resolution)
            zgrid = np.repeat(30,grid_resolution**2)
            grid_dimensions = np.stack((xgrid,ygrid,zgrid),axis=1)
            
            # Manually calculate the TDOAs for each microphone-pair using function TDOA_grid
            gridTDOA = localization.TDOA_grid(grid_dimensions)
            
            # Compare the calculated TDOA with the measured TDOA and find the point where their difference is the smallest
            comparison = np.linalg.norm(gridTDOA - tdoa, axis=1)
            best = grid_dimensions[np.argmin(comparison)]
            
            # To make the algorithm more accurate, once a point has been found, the algorithm will be looped
            # Set the dimensions for a new grid to have a higher resolution around the found point
            if i<finetuning:
                crop = 2/(grid_resolution**(i+1))
                min_x = best[0] - 460*crop
                max_x = best[0] + 460*crop
                min_y = best[1] - 460*crop
                max_y = best[1] + 460*crop
        return best