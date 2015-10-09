# Determine the Sampling Rate (file samples are 300ms speech)
#   Note: there are 2400 samples.

s = []

with open("lab.dat") as labdat:
    for line in labdat.readlines():
        if line != "\n": s.append(line.rstrip())

sampling_rate = len(s)/0.3 # 0.3s...


# Apply the ideal delay operator with delay 5ms, 10ms, and 15ms
y = [[], [], []]  # For the ideal delay operators at 5, 10 and 15.

def ideal_delay_by_5ms(input_signal):
    output_signal = []
    # ideal delay y[n] = s[n-n0]
    n0 = len(input_signal) / (5 * samp_rate)
    for n in range(0, len(input_signal)):
        if n < n0:
            output_signal.append(0)
        else:
            output_signal.append(input_signal(n-n0))
    return output_signal


# Apply the moving average with k1=k2=5, 10, and 15ms

# Convolve the signal with a window of length 10ms

# Extract the shirt0term energy signal from the signal in labrotory.dat

# Plot the orginal signal, the energy signal, the magnitude signal and the .ZCR signal as a function of time


