
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse
from IPython.display import Audio
from refsignal import refsignal            # model for the EPO4 audio beacon signal
from wavaudioread import wavaudioread
from recording_tool import recording_tool


def __init__(self, recording, debug=False):
    # Store the recordings
    # Load the reference signal from memory
    # x_car, y_car = self.localization()
    
def localization(self):
    # Split each recording into individual pulses
    # Calculate TDOA between different microphone pairs
    # Run the coordinate_2d using the calculated TDOAs
    
def TDOA(self, rec1, rec2):
    # Calculate channel estimation of each recording using ch2 or ch3
    # Calculate TDOA between two recordings based on peaks
    # in the channel estimate
    @staticmethod
    
def ch3(x, y):
    # Channel estimation
    print()
def coordinate_2d(self, D12, D13, D14):
    # Calculate 2D coordinates based on TDOA measurements
    # using the linear algebra given before
    print()
if __name__ == "__main__":
    # Main block for testing
    # Read the .wav file
    # Localize the sound source
    # Present the results
    print()