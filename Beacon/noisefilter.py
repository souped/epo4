import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

def autocorrelation(seq):
    n = len(seq)
    result = np.correlate(seq, seq, mode='full')
    return result[n-1:]

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Beacon specs
audio_file = "Beacon/reference6.wav"

# Read the audio file
Fs, audio = wavfile.read(audio_file)

# Check if the audio has more than one channel and select the second channel if it does
if audio.ndim > 1:
    audio = audio[:, 1]

# Extract the segment from reference6.wav
segment = audio[17500:20000]

# Apply low-pass filter to the segment
cutoff_frequency = 1000  # Cutoff frequency in Hz
filtered_segment = lowpass_filter(segment, cutoff_frequency, Fs)

# Plot the original and filtered segments
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(segment)
plt.title("Original Signal Segment for Beacon/reference6.wav")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(filtered_segment)
plt.title("Filtered Signal Segment for Beacon/reference6.wav")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.savefig("reference6_segment_filtered.png")
plt.close()

# Compute the autocorrelation of the filtered segment
autocorr = autocorrelation(filtered_segment)

# Plot the autocorrelation of the filtered segment
plt.figure()
plt.plot(autocorr, marker='o', linestyle='-', color='b')
plt.title('Autocorrelation of Filtered Beacon/reference6.wav Segment')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig("reference6_autocorr_filtered.png")
plt.close()

print("Plots have been saved as 'reference6_segment_filtered.png' and 'reference6_autocorr_filtered.png'")
