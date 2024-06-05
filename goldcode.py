import numpy as np
import matplotlib.pyplot as plt

def generate_m_sequence(n, feedback_taps):
    """Generate an m-sequence (maximum length sequence)"""
    m_seq = np.zeros((2**n - 1,), dtype=int)
    state = np.ones(n, dtype=int)  # Initial state
    for i in range(2**n - 1):
        m_seq[i] = state[-1]
        new_bit = np.bitwise_xor.reduce(state[feedback_taps])
        state = np.roll(state, 1)
        state[0] = new_bit
    return m_seq

def xor_sequences(seq1, seq2):
    """XOR two sequences to generate a Gold code"""
    return np.bitwise_xor(seq1, seq2)

def autocorrelation(seq):
    """Compute the autocorrelation of a sequence"""
    n = len(seq)
    result = np.correlate(seq, seq, mode='full')
    return result[n-1:]

def binary_to_hex(binary_string):
    """Convert binary string to hexadecimal string"""
    # Pad the binary string with leading zeros to make it a multiple of four bits
    while len(binary_string) % 4 != 0:
        binary_string = '0' + binary_string
    # Convert binary to hexadecimal
    hex_string = hex(int(binary_string, 2))[2:]
    return hex_string.upper()  # Convert to uppercase for consistency


# Parameters
n = 5  # Length parameter for m-sequences
feedback_taps_1 = [4]  # Feedback taps for the first m-sequence
feedback_taps_2 = [3, 4]  # Feedback taps for the second m-sequence

# Generate m-sequences
m_seq1 = generate_m_sequence(n, feedback_taps_1)
m_seq2 = generate_m_sequence(n, feedback_taps_2)

# Generate Gold code by XORing m-sequences
gold_code = xor_sequences(m_seq1, m_seq2)

# Convert Gold code to binary string
binary_gold_code = ''.join(map(str, gold_code))
if len(binary_gold_code) % 4 != 0:
    binary_gold_code = '0' + binary_gold_code

# Compute autocorrelation
autocorr = autocorrelation(gold_code)

# Convert binary Gold code to hexadecimal
hex_gold_code = binary_to_hex(binary_gold_code)

print("Gold code (binary):", binary_gold_code)
print("Gold code (hexadecimal):", hex_gold_code)
print("length: ", len(binary_gold_code))

# Plot autocorrelation
plt.figure()
plt.plot(autocorr, marker='o', linestyle='-', color='b')
plt.title('Autocorrelation of Gold Code')
plt.xlabel('Shift')
plt.ylabel('Autocorrelation')
plt.savefig("goldcode2")
plt.close()
