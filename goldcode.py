import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

def lfsr(seed, taps, length):
    """Generate an m-sequence using an LFSR with specified taps."""
    sr = seed
    xor = 0
    result = []
    for _ in range(length):
        result.append(sr[-1])
        for t in taps:
            xor ^= sr[t-1]
        sr = [xor] + sr[:-1]
        xor = 0
    return np.array(result)

def generate_gold_codes(mseq1, mseq2):
    """Generate Gold codes from two m-sequences."""
    gold_codes = []
    for i in range(len(mseq1)):
        gold_codes.append(np.bitwise_xor(mseq1, np.roll(mseq2, i)))
    gold_codes.append(mseq1)
    gold_codes.append(mseq2)
    return gold_codes

def evaluate_codes(gold_codes):
    """Evaluate the autocorrelation and cross-correlation of Gold codes."""
    num_codes = len(gold_codes)
    codes_per_page = 16
    num_pages = (num_codes + codes_per_page - 1) // codes_per_page

    for page in range(num_pages):
        plt.figure(figsize=(15, 8))
        start_idx = page * codes_per_page
        end_idx = min((page + 1) * codes_per_page, num_codes)
        
        for i, gold_code in enumerate(gold_codes[start_idx:end_idx], start=start_idx):
            autocorr = correlate(gold_code, gold_code, mode='full')
            lags = correlation_lags(len(gold_code), len(gold_code))
            
            
            plt.subplot(4, 4, i - start_idx + 1)
            plt.plot(lags, autocorr)  # Shift the lags to center at 0
            plt.title(f'Autocorrelation of Gold Code {i}')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.grid(True)

        plt.tight_layout()
        plt.show()



def gold_code_to_hex(gold_code):
    """Convert a binary Gold code to a hexadecimal string."""
    binary_string = ''.join(str(bit) for bit in gold_code)
    hex_string = hex(int(binary_string, 2))[2:]  # Remove '0x' prefix
    return hex_string.upper()  # Upper case for consistencyS

# Example seeds and taps for two m-sequences
seed1 = [0, 0, 0, 0, 1]
seed2 = [0, 0, 1, 0, 1]
taps1 = [5, 2]
taps2 = [5, 3]

# Generate m-sequences
mseq1 = lfsr(seed1, taps1, 2**5 - 1)
mseq2 = lfsr(seed2, taps2, 2**5 - 1)

# Generate Gold codes
gold_codes = generate_gold_codes(mseq1, mseq2)

# Print all Gold codes in hexadecimal format
for i, gold_code in enumerate(gold_codes):
    hex_code = gold_code_to_hex(gold_code)
    print(f"Gold Code {i} in Hex: {hex_code}")

# Evaluate Gold codes
evaluate_codes(gold_codes)
