from numpy import mean

# Determine the Sampling Rate (file samples are 300ms speech)
#   Note: there are 2400 samples.

s = []

with open("lab.dat") as labdat:
    for line in labdat.readlines():
        s.append(line.rstrip())

sample_rate = len(s)/0.3 # 0.3s...

# Apply the ideal delay operator with delay 5ms, 10ms, and 15ms
y = [[], [], []]  # For the ideal delay operators at 5, 10 and 15.

# ideal delay y[n] = s[n-n0]
def ideal_delay_by_5ms(input_signal):
    output_signal = []
    n0 = sample_rate * 0.005  # how many signals in 5 msecs?
    for n in range(0, len(input_signal)):
        if n < n0:
            output_signal.append(0)
        else:
            output_signal.append(input_signal[int(n-n0)])  # We can int() the indice as n-n0 always an integer
    return output_signal


y[0] = ideal_delay_by_5ms(s)     # 5ms
y[1] = ideal_delay_by_5ms(y[0])  # 10ms
y[2] = ideal_delay_by_5ms(y[1])  # 15ms

# Apply the moving average with k1=k2=5, 10, and 15ms
# Three cases to be aware of, as labelled in the code:
#   - 1: Moving average is clipped as there's too few measurements to go backward at the beginning of the signal
#   - 2: Ordinary moving average function
#   - 3: Moving average is clipped as there's too few measurements to go forward at the end of the signal
def moving_average(k1, k2, input_signal):
    output_signal = []
    #coefficient = 1/(k1 + k2 + 1)  # Shouldn't need this as numpy calculating for us
    for i in range(0, k1):
        output_signal[i] = mean(input_signal[0, i])
    for i in range(k1, len(input_signal) - k2):
        output_signal[i] = mean(input_signal[i-k1, i+k2])
    for i in range(len(input_signal) - k2, k2):
        output_signal[i] = mean(input_signal[i-k2, len(input_signal)])
    
    return output_signal
    
def moving_average_lab(k, input_signal):
    return moving_average(k, k, input_signal)

m_a = [[], [], []]
for i in range(0, 3):
    m_a[i] = moving_average_lab((i+1)*5, s)

# Convolve the signal with a window of length 10ms

# Extract the short-term energy signal from the signal in labrotory.dat

# Plot the orginal signal, the energy signal, the magnitude signal and the .ZCR signal as a function of time


