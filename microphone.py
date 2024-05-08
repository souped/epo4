# displays basic communication with microphone

import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
from scipy.io import wavfile

"""
eigenlijk zouden al deze functies nog in een Class moeten 
zodat je de globals niet meer global hoeft te maken
"""

# globals?
channelnumbers = 1
Fs = 48000
seconds = 2
N = int(Fs * seconds)

def list_devices():
    """Lists all audio devices and 
    returns index if devicename == "AudioBox 1818 VSL" else 0
    """
    pyaudio_handle = pyaudio.PyAudio()

    for i in range(pyaudio_handle.get_device_count()):
        device_info = pyaudio_handle.get_device_info_by_index(i)
        print(f"{i}, name: {device_info['name']}, inputchannels: {device_info['maxInputChannels']}")
        if device_info['name'] == "AudioBox 1818 VSL":
            pyaudio_handle.terminate()
            return i  
    pyaudio_handle.terminate()
    return 0

def seperate(data, channels = channelnumbers):
    """lists audio per channel
    
    returns -- list of channels
    [[channel1],
    [channel2],
    [channel3], ...]"""
    result = []
    for i in range(channels):
        channel = [k for i,k in enumerate(data[i:]) if i%channels == 0]
        result.append(channel)
    return np.array(result)

def record_audio(N, devidx, channels = channelnumbers, Fs = Fs):
    print(f"Recording for {(N / Fs):.2f} seconds")
    pyaudio_handle = pyaudio.PyAudio()
    stream = pyaudio_handle.open(input_device_index=devidx,
    channels=channels,
    format=pyaudio.paInt16,
    rate=Fs,
    input=True)
    
    samples = stream.read(N)
    data = np.frombuffer(samples, dtype='int16')
    pyaudio_handle.terminate()
    return seperate(data)

def write_wavfile(audio, filename = "wavfile.wav"):
    """writes multichannel audio in np.Array to .wav file.

    audio -- numpy array in shape (Nframes, Nchannels)"""
    audio = audio.T
    wavfile.write(filename, Fs, audio.astype(np.int16))

def read_wavfile(filename):
    """Reads and plots a wavefile. Requires plt.show() to be called seperately."""
    samplerate, audioBuffer = wavfile.read(filename)
    # audiobuffer of shape (Nsamples, Nchannels)
    # set up time axis
    time = np.arange(0, len(audioBuffer)/samplerate , 1/samplerate)
    ax = plt.figure().subplots()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("magnitude [??]") # ???
    lines = ax.plot(time, audioBuffer)
    # add labels to each channel plot.
    for i, line in enumerate(lines):
        line.set_label(f"channel {i+1}")
    ax.legend()

if __name__ == "__main__":
    device_index = list_devices()
    print(f"device index used: {device_index}")
    audio = record_audio(N, device_index)
    write_wavfile(audio)