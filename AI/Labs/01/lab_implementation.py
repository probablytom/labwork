from numpy import mean

# Determine the Sampling Rate (file samples are 300ms speech)
#   Note: there are 2400 samples.

s = []

with open("lab.dat") as labdat:
    for line in labdat.readlines():
        s.append(line.rstrip())

sample_rate = len(s)/0.3 # 0.3s sample length

def time_to_samples(millis):
    return millis*sample_rate

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
    moving_average_window = time_to_samples( (i+1)*5 )
    m_a[i] = moving_average_lab(moving_average_window, s)



# Convolve the signal with a window of length 10ms
def convolve(input_signal, window=10):
    output_signal = []
    window = time_to_samples(window)

    for i in range(0, window):
        output_signal[i] = sum(input_signal[0, i])
    for i in range(window, len(input_signal)):
        output_signal[i] = sum(input_signal[i-window, i])

    return output_signal

convolved_signal = convolve(s)

# Extract the short-term energy signal from the signal in laboratory.dat

# Helper function for signs and things
def sign(n):
    return n >= 0  # We can do this because Python treats True and False as 1 and 0 respectively.

def sign_difference(a, b):
    return abs( (sign(a)-sign(b))/2 )

# Helper function for general averaged convolution stuff
def averaged_convolution(input_signal, modifier_function = lambda x: x, window=30):
    output_signal = []
    input_signal = map(modifier_function, input_signal)
    window = time_to_samples(window)

    for i in range(0, window):
        output_signal[i] = sum(map(lambda x: x/i, input_signal[0, i]))

    input_signal = map(lambda x: x/window, input_signal)

    for i in range(window, len(input_signal)):
        output_signal[i] = sum(input_signal[i-window, i])

short_term_energy_signal = averaged_convolution(lambda x: x**2, s)
magnitude = averaged_convolution(abs, s)

sign_difference_signal = [0]
for i in range(1, len(s)):
    sign_difference_signal[i] = sign_difference(s[i], s[i-1])

zcr = averaged_convolution(sign_difference_signal)

# Plot the original signal, the energy signal, the magnitude signal and the .ZCR signal as a function of time


