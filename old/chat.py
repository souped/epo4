import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import find_peaks, medfilt
import os

class Localization:
    def __init__(self, debug=False):
        self.debug = debug
    
    def localization(self, audiowav, Fs):
        audio_channels = [audiowav[:, i] for i in range(audiowav.shape[1])]
        Fref, ref_signal = wavfile.read("opnames/reference.wav")
        ref_signal = ref_signal[:, 0]

        segments_channels = [self.detect_segments(channel) for channel in audio_channels]
        segments_channels = [segments[5:35] for segments in segments_channels]

        ref_segments = self.detect_segments(ref_signal)
        ref = ref_segments[12][750:1500]

        if self.debug:
            self.plot_signal(audio_channels[0], "Input audio", "audio")
            self.plot_signal(segments_channels[0][5], "Input segment 5", "seg_5")

        channel_responses = [self.estimate_channel_responses(segments, ref) for segments in segments_channels]

        if self.debug:
            self.plot_signal(channel_responses[1][5], "ch1", "ch1_5")

        peaks_channels = [self.find_segment_peaks(responses) for responses in channel_responses]
        
        # Debug: Print and plot peak values
        for i, peaks in enumerate(peaks_channels):
            print(f"Peaks for channel {i+1}: {peaks}")
            self.plot_peaks(channel_responses[i], peaks, f"Channel {i+1} Peaks", f"channel_{i+1}_peaks")

        mean_peaks = [np.mean(np.sort(peaks)[10:-10]) for peaks in peaks_channels]

        # Debug: Print mean peak values
        print(f"Mean peaks: {mean_peaks}")

        TDOA12 = self.TDOA(mean_peaks[0], mean_peaks[1], Fs)
        TDOA13 = self.TDOA(mean_peaks[0], mean_peaks[2], Fs)
        TDOA14 = self.TDOA(mean_peaks[0], mean_peaks[3], Fs)

        # Debug: Print TDOA values
        print(f"TDOA12: {TDOA12}, TDOA13: {TDOA13}, TDOA14: {TDOA14}")

        x, y = self.coordinate_2d(TDOA12, TDOA13, TDOA14)
        
        return x, y

    def detect_segments(self, audio_signal):
        num_segments = 40
        segment_length = len(audio_signal) // num_segments
        segments = [abs(audio_signal[i*segment_length : (i+1)*segment_length]) for i in range(num_segments)]
        
        # Apply median filter to each segment to reduce noise
        filtered_segments = [medfilt(segment, kernel_size=5) for segment in segments]
        
        # Debug: Print segments and plot them
        if self.debug:
            for i, segment in enumerate(filtered_segments):
                print(f"Segment {i+1}: {segment}")
                self.plot_signal(segment, f"Segment {i+1}", f"segment_{i+1}")

        return filtered_segments

    def find_segment_peaks(self, segment_signal):
        peaks_list = [np.argmax(segment) for segment in segment_signal]
        
        # Debug: Print peaks
        if self.debug:
            for i, peak in enumerate(peaks_list):
                print(f"Peak {i+1}: {peak}")

        return peaks_list
    
    def TDOA(self, peak1, peak2, Fs):
        mean_tdoa = (peak2 - peak1) / Fs * 343
        return mean_tdoa

    @staticmethod
    def estimate_channel_responses(segments, ref):
        responses = [Localization.ch3(segment, ref) for segment in segments]
        
        # Debug: Print channel responses and plot them
        for i, response in enumerate(responses):
            print(f"Channel response {i+1}: {response}")
            Localization.plot_signal(response, f"Channel Response {i+1}", f"channel_response_{i+1}")

        return responses

    @staticmethod
    def ch3(signal_1, reference_signal):
        L = len(signal_1) - len(reference_signal) + 1
        Lhat = max(len(reference_signal), len(signal_1))
        epsi = 0.005

        reference_signal = np.append(reference_signal, [0] * (L-1))
        fft_signal_1 = fft(signal_1)
        fft_reference_signal = fft(reference_signal)
        ii = (np.abs(fft_reference_signal)) < (np.max(np.abs(fft_reference_signal)) * epsi)

        H = np.divide(fft_signal_1, fft_reference_signal, where=~ii, out=np.zeros_like(fft_signal_1))
        h = np.real(ifft(H))
        h = h[:Lhat]

        # Debug: Print deconvolved signal
        print(f"Deconvolved signal: {h}")

        return abs(h)

    @staticmethod
    def coordinate_2d(D12, D13, D14):
        D23 = D13 - D12
        D24 = D14 - D12
        D34 = D14 - D13

        X1, X2, X3, X4 = np.array([0, 0]), np.array([0, 4.8]), np.array([4.8, 4.8]), np.array([4.8, 0])

        B = np.array([
            [D12**2 - np.linalg.norm(X1)**2 + np.linalg.norm(X2)**2],
            [D13**2 - np.linalg.norm(X1)**2 + np.linalg.norm(X3)**2],
            [D14**2 - np.linalg.norm(X1)**2 + np.linalg.norm(X4)**2],
            [D23**2 - np.linalg.norm(X2)**2 + np.linalg.norm(X3)**2],
            [D24**2 - np.linalg.norm(X2)**2 + np.linalg.norm(X4)**2],
            [D34**2 - np.linalg.norm(X3)**2 + np.linalg.norm(X4)**2]
        ])

        A = np.array([
            [2*(X2-X1)[0], 2*(X2-X1)[1], -2 * D12, 0, 0],
            [2*(X3-X1)[0], 2*(X3-X1)[1], 0, -2 * D13, 0],
            [2*(X4-X1)[0], 2*(X4-X1)[1], 0, 0, -2 * D14],
            [2*(X3-X2)[0], 2*(X3-X2)[1], 0, -2 * D23, 0],
            [2*(X4-X2)[0], 2*(X4-X2)[1], 0, 0, -2 * D24],
            [2*(X4-X3)[0], 2*(X4-X3)[1], 0, 0, -2 * D34]
        ])

        A_inv = np.linalg.pinv(A)
        result = np.dot(A_inv, B)
        x, y = result[0, 0], result[1, 0]

        # Debug: Print coordinate calculation intermediate values
        print(f"A: {A}, B: {B}, A_inv: {A_inv}, result: {result}")

        return x, y

    @staticmethod
    def plot_signal(signal, title, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(signal, color='C0')
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_peaks(signal, peaks, title, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(signal, color='C0')
        plt.scatter(peaks, signal[peaks], color='red')
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    localizer = Localization(debug=True)
    Fs, ABS5 = wavfile.read("opnames/record_x150_y185.wav")
    x_car5, y_car5 = localizer.localization(ABS5, Fs)
    print(f"Coordinates_x150_y185 : x = {x_car5}, y = {y_car5}")
