import numpy as np
import matplotlib.pyplot as plt

def refsignal(Ncodebits, Timer0, Timer1, Timer3, code, Fs):
    """
    Input: Ncodebits, Timer0, Timer1, Timer3, code, as for the AVR
        Extension: if Timer0 == -1, then no carrier modulation
        Fs: sample rate at which to generate the template (e.g., 40e3)

    The default parameters of the audio beacon are obtained using
        x = refsignal(32, 3, 8, 2, '0x92340f0faaaa4321', Fs);

    Output:
        x: the transmitted signal (including the silence period)
        last: the last sample before the silence period
    """

    # First perform sanity checks on the input
    if not isinstance(Ncodebits, int): raise TypeError("Ncodebits must be an integer")
    if not isinstance(Timer0, int): raise TypeError("Timer0 must be an integer")
    if not isinstance(Timer1, int): raise TypeError("Timer1 must be an integer")
    if not isinstance(Timer3, int): raise TypeError("Timer3 must be an integer")
    if not isinstance(code, int): raise TypeError("code must be a hex string")
    if not isinstance(Fs, int): raise TypeError("Fs must be an integer")

    # Lists to match Timerx to frequencies (Hz)
    FF0 = [i * 10 ** 3 for i in [0, 5, 10, 15, 20, 25, 30]]
    FF1 = [i * 10 ** 3 for i in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]]
    FF3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Compute the corresponding frequencies (Hz)
    f0 = FF0[Timer0 + 1]  # (also allow for '-1' as input)
    f1 = FF1[Timer1]
    f3 = FF3[Timer3]

    # Convert hex code string into binary string
    bincode = f'{code:0>42b}'

    # Generate template
    Nx = round(Fs / f3)  # Number of samples in template vector (integer)
    x = np.zeros(Nx)

    Np = Fs / f1  # Number of samples of one "Timer1" period (noninteger)

    for i in range(1, Ncodebits + 1):
        index = np.arange(round((i - 1) * Np + 1), round(i * Np) + 1)
        bit = int(bincode[i - 1])
        x[index - 1] = np.ones(len(index)) * bit

    # Modulate x on a carrier with frequency f0
    carrier = np.cos(2 * np.pi * f0 / Fs * np.arange(0, Nx))
    xmod = np.round(carrier + 1)  # Convert sine wave to block pulses

    x = np.multiply(x, xmod)

    # Compute location of last nonzero sample
    last = round(Ncodebits * Np) - 1

    return x, last

ref = refsignal(Ncodebits=32, Timer0=5, Timer1=50, Timer3=500, code=, Fs=44100)

print(ref)



