import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

def lfsr(seed, taps, length):
    """Generate an m-sequence using an LFSR with specified taps."""
    sr = seed[:]
    result = []
    for _ in range(length):
        output = sr[-1]
        result.append(output)
        feedback = 0
        for tap in taps:
            feedback ^= sr[tap-1]
        sr.pop()
        sr.insert(0, feedback)
    return np.array(result)

def generate_gold_code(mseq1, mseq2):
    """Generate Gold code from two m-sequences."""
    gold_code = np.bitwise_xor(mseq1, mseq2)
    return gold_code

def evaluate_gold_code(gold_code):
    """Evaluate the autocorrelation of a single Gold code."""
    autocorr = correlate(gold_code, gold_code, mode='full')
    lags = correlation_lags(len(gold_code), len(gold_code))

    plt.figure(figsize=(10, 6))
    plt.plot(lags, autocorr)
    plt.title('Autocorrelation of Gold Code')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.savefig("gold_codes\\correlation\\EA28B7C5")
    plt.grid(True)
    plt.close()

def gold_code_to_binary(hex_code):
    """Convert a hexadecimal Gold code to binary numpy array."""
    binary_string = bin(int(hex_code, 16))[2:].zfill(32)
    return np.array([int(bit) for bit in binary_string])

# Example hexadecimal Gold code
hex_code = 'EA28B7C5'

# Convert hexadecimal to binary
gold_code_binary = gold_code_to_binary(hex_code)

# Example seeds and taps for two m-sequences (you may need to adjust taps as per your LFSR configuration)
seed1 = [0, 0, 0, 0, 1]
seed2 = [0, 0, 1, 0, 1]
taps1 = [5, 2]
taps2 = [5, 3]

# Generate m-sequences
mseq1_generated = lfsr(seed1, taps1, len(gold_code_binary))
mseq2_generated = lfsr(seed2, taps2, len(gold_code_binary))

# Generate Gold code
gold_code_generated = generate_gold_code(mseq1_generated, mseq2_generated)

# Evaluate Gold code
evaluate_gold_code(gold_code_generated)
