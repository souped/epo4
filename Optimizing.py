
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import numpy as np
import time
from microphone import Microphone
Fs = 48000

class localization:
    def localization(audiowav, ref):
        # function goal: return the coorinates (x, y) using a ref signal and the audiosignal
        TDOA_list = []  
        
        #loop the calculateions so it will be done for a microphone and the following one, creating 10 microphone pairs (12, 13, 14, ... , 45)
        for i in range(5):
            for j in range(i+1, 5):
                # take one channel of the audio signal
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
        return x_car, y_car #return the coordinates
    
    def process_channel(channel_data, ref):
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
    
    def coordinates_2d(tdoa,min_x=0,max_x=460,min_y=0,max_y=460,size=5,finetuning=5):     
        #definition goal: return the coordinates of the car using the measured and calculated TDOA's  
        for i in range(finetuning):
            
            #set the grid dimensions using the given boundaries
            xgrid = np.tile(np.linspace(min_x,max_x,size+2)[1:-1],size)
            ygrid = np.repeat(np.linspace(min_y,max_y,size+2)[1:-1],size)
            zgrid = np.repeat(30,size**2)
            grid_dimensions = np.stack((xgrid,ygrid,zgrid),axis=1)
            
            #manually calculate the TDOA's for each microphone-pair using the function gridTDOA
            gridTDOA = localization.TDOA_grid(grid_dimensions)
            
            #compare the calculated TDOA with the measure TDOA and find the point where their difference is the smallest
            comparison = np.linalg.norm(gridTDOA - tdoa, axis=1)
            best = grid_dimensions[np.argmin(comparison)]
            
            #To make the algorithm more accurate, once a point has been found, the algorithm will be looped
            #set the dimensions for a new grid to have a higher resolution around the found point (same gridpoints will be used for a smaller area)
            if i<finetuning:
                padding = 460/(size**(i+1))/2
                min_x = best[0] - padding
                max_x = best[0] + padding
                min_y = best[1] - padding
                max_y = best[1] + padding
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
        "gold_codes/gold_code_ref13.wav"]
    
    audio_files = [
        "gold_codes/gold_code13_test200-195.wav",
        "gold_codes/gold_code13_test128-375.wav",
        "gold_codes/gold_code13_test334-354.wav"
    ]


    for ref in ref_files:
        Fs, audio = wavfile.read(ref)
        
        audio = audio[:, 0]
        for i,val in enumerate(audio):
            if val > 50:
                start = i
                break

        pulse_length = 306

        print(f"segment index range: {start}, {start+pulse_length}")
        segment = audio[start:start+pulse_length]
    
        start = time.time()
        for file in audio_files:
            Fs, audio = wavfile.read(file)
            x_car, y_car = localization.localization(audio, segment)
            print(f"{file}, {ref}: x = {x_car:.4f}, y = {y_car:.4f}")
        print(" ")
    end = time.time()
    print("Total time:", end - start)

if __name__ == "__main_99_":
    localizer = localization()

    Fref, ref_signal = wavfile.read("reference6.wav")
    ref_signal =  ref_signal[:,1]
    ref = ref_signal[18800:19396]

    #RECORD AUDIO 
    mic = Microphone(channelnumbers = 1, Fs= 48000)
    device_index = Microphone.list_devices()
    print(f"device index used: {device_index}")
    seconds = 0.1
    # print(mic.record_audio(seconds, device_index)[0])
    fakes = [mic.record_audio(seconds, device_index)[0] for j in range(8)]
    fake = np.stack(fakes)
    print(f"fake: {fake.shape}")

    x_car, y_car = localization.localization(fake, ref)
    print(f"x = {x_car}, y = {y_car}")

def peak_idxs(channel):
    # peak_idx = np.argmax(audio_list)
    # print(audio_list[peak_idx])
    # return audio_list[peak_idx-5:peak_idx+70]

    # create list of peaks in this channel, peak is defined as every value greater than
    # max(channel) * 0.95
    m_i = np.argmax(channel)
    m = channel[m_i]
    a = 0.96
    th = m * a
    peaks_i = []
    for i,val in enumerate(channel):
        if val < th:
            continue
        else:
            peaks_i.append(i)
    print(f"max: {m}, th: {th}, list: ", end="")
    print(peaks_i)
    print(np.diff(peaks_i, 1))
    print(channel[peaks_i])

    # list of peaks indices made, now go to other channels
    return peaks_i

def segment_from_rec(audio):
    # audio is 5 channel array
    seglen = 12000
    seg = audio[:seglen*1.2]

        
# volgorde:
# call localization(audio, ref)
# split in channels, neem 2 channels
# process_channel(channel, ref) -> mean van de peaks
# in proc: detect_segments: snijd op in stukjes met 1 peak per stukjes
# 

if __name__ == "__main_33_":
    localizer = localization()
    start = time.time()
    Fref, ref_signal = wavfile.read("gold_codes/gold_code_ref13.wav")
    # for ch in ref_signal.T[:5]:
    #     test_fn(ch)
    idxs = peak_idxs(ref_signal.T[0])
    plt.plot(ref_signal.T[0])
    # plt.xlim([idxs[0]-550, idxs[0]+550])
    plt.show()
    


if __name__ == "__main_9_":
# Main block for testing
# Read the .wav file
# Localize the sound source
# Present the results
    localizer = localization()

    start=time.time()
    
    for i in range(1,13):
        # Fref, ref_signal = wavfile.read(f"gold_codes/gold_code_ref{i}.wav")
        pass

    Fref, ref_signal = wavfile.read("gold_codes/gold_code_ref4.wav")
    # Fref, ref_signal = wavfile.read("ref_sigs/reference6.wav")
    print(ref_signal.shape)
    ref =  ref_signal[5600:6200]
    # ref = test_fn(ref_signal)
    # ref = ref_signal[18800:19396]
    
    plt.plot(ref)
    plt.savefig("refsignalhuidig")
    plt.close()

    
    audio_files = [
        "vanafxy-20-230.wav",
        "vanafx-y-66-60.wav",
        "posx175posy110.wav"
    ]

    start = time.time()
    for file in audio_files:
        
        Fs, audio = wavfile.read(f"ref_sigs/{file}")
        print(f"fs: {Fs}")
        plt.plot(audio)
        plt.title(f"Reference signal for {file}")
        plot_filename = file.replace("Beacon/", "").replace(".wav", ".png")
        plt.savefig(plot_filename)
        plt.close()
        
        x_car, y_car = localization.localization(audio, ref)
        print(f"{file}: x = {x_car}, y = {y_car}")

    end = time.time()
    print("Total time:", end - start)

if __name__ == "__main__00":
    Fs, audio = wavfile.read(f"ref_sigs/goldcode13recordingtest.wav")
    axs = plt.figure().subplots(8)
    for channel in audio.T:
        plt.figure().subplots().plot(channel)
    plt.show()
