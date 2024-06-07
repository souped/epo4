import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
from scipy.io import wavfile

class Controller:
    pass

class Microphone:
    def __init__(self, cntrlr, channelnumbers: int=8, Fs: int=48000):
        #, seconds=6
        self.channelnumbers = channelnumbers
        self.Fs = Fs
        self.controller = cntrlr
        # self.N = int(self.Fs * seconds)
        self.handle = pyaudio.PyAudio()

    def list_devices():
        """Lists all audio devices and 
            returns index if devicename == "AudioBox 1818 VSL" else 0
        """
        pyaudio_handle = pyaudio.PyAudio()

        for i in range(pyaudio_handle.get_device_count()):
            device_info = pyaudio_handle.get_device_info_by_index(i)
            print(f"{i}, name: {device_info['name']}, inputchannels: {device_info['maxInputChannels']}")
            if device_info['name'] == "AudioBox 1818 VSL ":
                print("detected!")
                pyaudio_handle.terminate()
                return i  
        pyaudio_handle.terminate()
        return 0

    def separate(self, data, channels=None):
        """lists audio per channel
        
        returns -- list of channels
        [[channel1],
        [channel2],
        [channel3], ...]"""
        if not channels: channels = self.channelnumbers
        result = []
        for i in range(channels):
            channel = [k for i,k in enumerate(data[i:]) if i%channels == 0]
            result.append(channel)
        return np.array(result)
    
    def record_audio(self, seconds, devidx, channels = None, Fs = None):
        if not channels: channels = self.channelnumbers
        if not Fs: Fs = self.Fs
        N = int(Fs * seconds)
        print(f"Recording for {(N / Fs):.2f} seconds")
        stream = self.handle.open(input_device_index=devidx,
                                  channels=channels,
                                  format=pyaudio.paInt16,
                                  rate=Fs,
                                  input=True)
        # samples = stream.read(N)
        # data = np.frombuffer(samples, dtype='int16')
        while stream.is_active():
            time.sleep(0.1)
            print("still active...")
        
        stream.close()
        
        # return self.separate(data)
    
    def write_wavfile(self, audio, filename = "wavfile.wav"):
        """writes multichannel audio in np.Array to .wav file.

        audio -- numpy array in shape (Nframes, Nchannels)"""
        audio = audio.T
        wavfile.write(filename, self.Fs, audio.astype(np.int16))

    def read_wavfile(filename):
        """Reads and plots a wavefile.        
        Requires plt.show() to be called seperately."""
        samplerate, audioBuffer = wavfile.read(filename)
        # audiobuffer of shape (Nsamples, Nchannels)
        # set up time axis
        time = np.arange(0, len(audioBuffer)/samplerate , 1/samplerate)
        ax = plt.figure().subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude [??]") # ???
        lines = ax.plot(time, audioBuffer)
        # add labels to each channel plot.
        for i, line in enumerate(lines):
            line.set_label(f"channel {i+1}")
        ax.legend()

    def read_stream_callback_mode(self, seconds, devidx, channels = None, Fs = None):
        if not channels: channels = self.channelnumbers
        if not Fs: Fs = self.Fs
        N = int(Fs * seconds)

        pass

    def callback(in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype='int16')
        print(data)
        return (data, pyaudio.paContinue)

if __name__ == "__main__":
    cn = Controller()
    mic = Microphone(cn, channelnumbers=1)
    device_index = Microphone.list_devices()
    print(f"device index used: {device_index}")
    seconds = 2
    audio = mic.record_audio(seconds, device_index)
    mic.write_wavfile(audio)
    Microphone.read_wavfile("wavfile.wav")
    plt.show()
