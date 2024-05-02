# displays basic communication with microphone

import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
import wave
from scipy.io import wavfile

from dynamicplotter import DynamicPlotter

channelnumbers = 1
N = 100000
Fs = 48000

def list_devices():
    pyaudio_handle = pyaudio.PyAudio()

    for i in range(pyaudio_handle.get_device_count()):
        device_info = pyaudio_handle.get_device_info_by_index(i)
        print(i, device_info['name'])
        print(i, device_info['maxInputChannels'])
        if device_info['name'] == "AudioBox 1818 VSL":
            return i     
    return 0

def record_audio(N, devidx, Fs = 48000):
    print("recording...")
    pyaudio_handle = pyaudio.PyAudio()
    stream = pyaudio_handle.open(input_device_index=devidx,
    channels=channelnumbers,
    format=pyaudio.paInt16,
    rate=Fs,
    input=True)

    samples = stream.read(N)
    data = np.frombuffer(samples, dtype='int16')
    return data

def run_plotter(data, plotter):    
    for i in range(channelnumbers):
        channel = [k for i,k in enumerate(data[i:]) if i%channelnumbers == 0]
        # !!!note that its likely that i != microphone number 
        plotter.on_running(np.arange(0,len(channel)), channel, i)

def plot_single(data):
    ax = plt.figure().subplots()
    ax.plot(np.arange(0,len(data)), data)
    plt.show()

"""writes dualchannel audio in np.Array to .wav file"""
def write_to_file(audio):
    # Convert to (little-endian) 16 bit integers.
    audio = (audio * (2 ** 15 - 1)).astype("<h")

    with wave.open("wavfile.wav", "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(Fs)
        f.writeframes(audio.tobytes())

def read_wavfile(filename):
    samplerate, audioBuffer = wavfile.read(filename)
    time = np.arange(0, len(audioBuffer)/samplerate, 1/samplerate)
    print(audioBuffer.shape)
    plt.plot(time, audioBuffer)
    plt.show()


def clear_file(filename):
    with open(f"{filename}", "w") as f:
        f.write("")

if __name__ == "__main__":
    # plotter = DynamicPlotter()

    # dev_id = list_devices()

    # print("connecting...")
    # for i in range(10):
    #     data = record_audio(N, dev_id)
    #     process(data, plotter)
    #     print("plotted")
    #     time.sleep(0.3)
    
    testdata = record_audio(Fs*1, 0)
    testdata2 = record_audio(Fs*1, 0)
    audio = np.array([testdata,testdata2])
    write_to_file(audio=audio)
    read_wavfile("wavfile.wav")
    
    # plot_single(testdata)
    

    # xdata = []
    # ydata = []
    # for x in np.arange(0,10,0.5):
    #     xdata.append(x)
    #     ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
    #     plotter.on_running(xdata, ydata, 0)
    #     time.sleep(1)

    