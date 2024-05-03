# displays basic communication with microphone

import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
import wave
from scipy.io import wavfile

from dynamicplotter import DynamicPlotter

channelnumbers = 1
Fs = 48000
N = int(Fs * 2)

def list_devices():
    pyaudio_handle = pyaudio.PyAudio()

    for i in range(pyaudio_handle.get_device_count()):
        device_info = pyaudio_handle.get_device_info_by_index(i)
        print(f"{i}, name: {device_info['name']}, inputchannels: {device_info['maxInputChannels']}")
        if device_info['name'] == "AudioBox 1818 VSLLLLLL":
            return i     
    return 0

def record_audio(N, devidx, channels = 5, Fs = 48000):
    print("recording...")
    pyaudio_handle = pyaudio.PyAudio()
    stream = pyaudio_handle.open(input_device_index=devidx,
    channels=channels,
    format=pyaudio.paInt16,
    rate=Fs,
    input=True)

    samples = stream.read(N)
    data = np.frombuffer(samples, dtype='int16')
    return data

def run_plotter(data, plotter):    
    for i in range(channelnumbers):
        channel = [k for i,k in enumerate(data[i:]) if i%channelnumbers == 0]
        # note that its possible that i != microphone number 
        plotter.on_running(np.arange(0,len(channel)), channel, i)

def plot_single(data):
    ax = plt.figure().subplots()
    ax.plot(np.arange(0,len(data)), data)
    # plt.show()

def write_wavfile(audio, filename = "wavfile.wav"):
    """writes multichannel audio in np.Array to .wav file
    audio: numpy array in shape (Nframes, Nchannels)"""
    audio = audio.T
    wavfile.write(filename, Fs, audio.astype(np.int16))

def read_wavfile(filename):
    """Reads and plots a wavefile. Requires plt.show() to be called seperately."""
    samplerate, audioBuffer = wavfile.read(filename)
    # audiobuffer of shape (Nsamples, Nchannels)
    time = np.arange(0, len(audioBuffer)/samplerate , 1/samplerate)
    ax = plt.figure().subplots()
    lines = ax.plot(time, audioBuffer)
    # add labels to each channel plot.
    for i, line in enumerate(lines):
        line.set_label(f"channel {i+1}")
    ax.legend()

if __name__ == "__main__":
    # plotter = DynamicPlotter()

    dev_id = list_devices()
    
    for i in range(2):
        t1 = time.time()
        data = record_audio(N, dev_id, channelnumbers)
        t2 = time.time()
        print(f"time to record: {t2-t1}")

        t1 = time.time()
        write_wavfile(audio=data, filename=f"wavfile{i}.wav")
        t2 = time.time()
        print(f"time to write file: {t2-t1}")
        time.sleep(2)
    
    # testdata = record_audio(Fs*1, 0)
    # testdata2 = record_audio(Fs*1, 0)
    # audio = np.array([testdata,testdata2])
    
    for i in range(2):
        read_wavfile(f"wavfile{i}.wav")
    plt.show()

    # plot_single(testdata)
    

    # xdata = []
    # ydata = []
    # for x in np.arange(0,10,0.5):
    #     xdata.append(x)
    #     ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
    #     plotter.on_running(xdata, ydata, 0)
    #     time.sleep(1)

    