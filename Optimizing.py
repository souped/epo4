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
        # Split each recording into individual pulses
        TDOA_list=[]
        # Calculate TDOA between different microphone pairs
        for i in range(5):
            for j in range(i + 1,5):
                audio_channel_i=audiowav[:, i]
                audio_channel_j=audiowav[:, j]
                mean_peak_i=localization.process_channel(audio_channel_i,ref)
                mean_peak_j=localization.process_channel(audio_channel_j,ref)
                TDOA=localization.TDOA(mean_peak_j,mean_peak_i)
                TDOA_list.append(TDOA)
        location=localization.coordinates_2d(TDOA_list)
        x_car=location[0]
        y_car=location[1]
        return x_car,y_car

    def process_channel(channel_data,ref):
        segments=localization.detect_segments(channel_data)
        channel_responses=[localization.ch3(segment,ref) for segment in segments]
        channel_responses_array=np.array(channel_responses)
        peaks=localization.find_segment_peaks(channel_responses_array)
        sorted_peaks=np.sort(peaks)
        trimmed_peaks=sorted_peaks[1:-1]
        mean_peak=np.mean(trimmed_peaks)
        return mean_peak

    def detect_segments(audio_signal):
        segments=[]
        num_segments=8
        segment_length=len(audio_signal) // num_segments
        segments=[abs(audio_signal[i * segment_length: (i + 1) * segment_length]) for i in range(num_segments)]
        return segments

    def find_segment_peaks(segment_signal):
        peaks_list=[]
        for segment in segment_signal:
            peaks=np.argmax(segment)
            peaks_list.append(peaks)

        return peaks_list

    def TDOA(peak1,peak2):
        return (peak2 - peak1) / Fs

    @staticmethod
    def ch3(signal_1,reference_signal):
        Nsignal_1=len(signal_1)  # Length of x
        Nreference_signal=len(reference_signal)  # Length of y
        L=Nsignal_1 - Nreference_signal + 1  # Length of h
        Lhat=max(len(reference_signal),len(signal_1))
        epsi=0.005

        # Force x to be the same length as y
        reference_signal=np.append(reference_signal,[0] * (L - 1))  # Make x same length as y
        # print(len(reference_signal))

        # Deconvolution in frequency domain
        fft_signal_1=fft(signal_1)
        fft_reference_signal=fft(reference_signal)

        # Threshold to avoid blow ups of noise during inversion
        ii=(np.abs(fft_reference_signal)) < (np.max(np.abs(fft_reference_signal)) * epsi)

        H=np.divide(fft_signal_1,fft_reference_signal)
        H[ii]=0

        # H = [0 if condition else reference_signal for reference_signal, condition in zip(fft_signal_1, ii)]
        h=np.real(ifft(H))
        h=h[0:Lhat]

        return h

        # old way of coordinate derivation (Matrix), turned out to be too inaccurate

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

    def TDOA_grid(grid_dimensions):
        gridTDOA=[]
        mic_locs=np.array([[0,0,50],[0,460,50],[460,460,50],[460,0,50],[0,230,80]])
        for row in grid_dimensions:
            distances=[]
            for loc in mic_locs:
                dist=np.linalg.norm(loc - row)
                distances.append(dist)
            times=np.array(distances) / 34300
            TDOA=[]
            for i in range(0,len(times)):
                for j in range(i + 1,len(times)):
                    TDOA=np.append(TDOA,(times[i] - times[j]))
            gridTDOA=np.concatenate((gridTDOA,TDOA))
        gridTDOA=np.reshape(gridTDOA,(-1,10))
        return (gridTDOA)

    def coordinates_2d(tdoa,size=10,min_x=0,max_x=460,min_y=0,max_y=460,finetuning=5):
        for i in range(5):
            xgrid=np.tile(np.linspace(min_x,max_x,size + 2)[1:-1],size)
            ygrid=np.repeat(np.linspace(min_y,max_y,size + 2)[1:-1],size)
            zgrid=np.repeat(30,size ** 2)
            grid_dimensions=np.stack((xgrid,ygrid,zgrid),axis=1)

            gridTDOA=localization.TDOA_grid(grid_dimensions)

            error_list=np.linalg.norm(gridTDOA - tdoa,axis=1)
            best=grid_dimensions[np.argmin(error_list)]

            if i < finetuning:
                padding=460 / (size ** (i + 1)) / 2
                min_x=best[0] - padding
                max_x=best[0] + padding
                min_y=best[1] - padding
                max_y=best[1] + padding
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