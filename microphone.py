# displays basic communication with microphone

import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time

from dynamicplotter import DynamicPlotter


def list_devices():
    pyaudio_handle = pyaudio.PyAudio()

    for i in range(pyaudio_handle.get_device_count()):
        device_info = pyaudio_handle.get_device_info_by_index(i)
        print(i, device_info['name'])
        if device_info['name'] == "":
            return i     
    return 0

def record_audio(N, devidx, Fs = 48000):
    pyaudio_handle = pyaudio.PyAudio()
    stream = pyaudio_handle.open(input_device_index=devidx,
    channels=5,
    format=pyaudio.paInt16,
    rate=Fs,
    input=True)

    samples = stream.read(N)
    data = np.frombuffer(samples, dtype='int16')
    return data

def process(data, plotter):    
    for i in range(5):
        channel = [k for i,k in enumerate(data[i:]) if i%5 == 0]
        # !!!note that its likely that i != microphone number 
        plotter.on_running(channel, np.arange(len(channel)), i)
          

if __name__ == "__main__":
    plotter = DynamicPlotter()

    print("connecting...")
    dev_id = list_devices()
    data = record_audio(100, dev_id)
    process(data, plotter)

    xdata = []
    ydata = []
    for x in np.arange(0,10,0.5):
        xdata.append(x)
        ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
        plotter.on_running(xdata, ydata, 0)
        time.sleep(1)

    