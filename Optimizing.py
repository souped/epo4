
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import numpy as np
import time
from microphone import Microphone
Fs = 48000

class localization:
    #def __init__(recording, debug=False):
    def localization(audiowav, ref):
        # Split each recording into individual pulses
        TDOA_list = []
        # Calculate TDOA between different microphone pairs
        for i in range(5):
            for j in range(i+1, 5):
                audio_channel_i = audiowav[i,:]
                audio_channel_j = audiowav[j,:]
                mean_peak_i = localization.process_channel(audio_channel_i, ref)
                mean_peak_j = localization.process_channel(audio_channel_j, ref)
                TDOA = localization.TDOA(mean_peak_j, mean_peak_i)
                TDOA_list.append(TDOA)
        location = localization.coordinates_2d(TDOA_list)
        x_car = location[0]
        y_car = location[1]
        return x_car, y_car
    
    def process_channel(channel_data, ref):
        segments = localization.detect_segments(channel_data)
        channel_responses = [localization.ch3(segment, ref) for segment in segments]
        channel_responses_array = np.array(channel_responses)
        peaks = localization.find_segment_peaks(channel_responses_array)
        sorted_peaks = np.sort(peaks)
        trimmed_peaks = sorted_peaks[1:-1]
        mean_peak = np.mean(trimmed_peaks)
        return mean_peak

    def detect_segments(audio_signal):
        segments = []
        num_segments = 6
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
        return (peak2 - peak1)/Fs

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

        return h
    
    def TDOA_grid(grid_dimensions):
        gridTDOA = []
        mic_locs = np.array([[0,0,50],[0,460,50],[460,460,50],[460,0,50],[0,230,80]])
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
    
    def coordinates_2d(tdoa,size=10,min_x=0,max_x=460,min_y=0,max_y=460,finetuning=5): 
        for i in range(5):
            xgrid = np.tile(np.linspace(min_x,max_x,size+2)[1:-1],size)
            ygrid = np.repeat(np.linspace(min_y,max_y,size+2)[1:-1],size)
            zgrid = np.repeat(30,size**2)
            grid_dimensions = np.stack((xgrid,ygrid,zgrid),axis=1)

            gridTDOA = localization.TDOA_grid(grid_dimensions)

            error_list = np.linalg.norm(gridTDOA - tdoa, axis=1)
            best = grid_dimensions[np.argmin(error_list)]

            if i<finetuning:
                padding = 460/(size**(i+1))/2
                min_x = best[0] - padding
                max_x = best[0] + padding
                min_y = best[1] - padding
                max_y = best[1] + padding
        return best

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

def test_fn(audio_list):
    # peak_idx = np.argmax(audio_list)
    # print(audio_list[peak_idx])
    # return audio_list[peak_idx-5:peak_idx+70]
    # detect segment:
    pass



if __name__ == "__main__":
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