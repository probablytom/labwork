from numpy import mean, log

speech_samples = []

def import_input_data():
    global speech_samples
    speech_samples = []
    for i in range(1, 51):
        input_data = []
        filepath = "inputs/speech_"+str(i).zfill(2)+".dat"
        with open(filepath) as data:
            for line in data:
                input_data.append(int(line.rstrip()))
        speech_samples.append(input_data)
        input_data = []
        filepath = "inputs/silence_"+str(i).zfill(2)+".dat"
        with open(filepath) as data:
            for line in data:
                input_data.append(int(line.rstrip()))
        speech_samples.append(input_data)

import_input_data()
sample_rate = []
for i in range(1, len(speech_samples) + 1):
    sample_rate.append( int( len(speech_samples[i-1])/0.3 ) )  # Sample is always 0.3 seconds long. 

# Just as a helper function:
def time_to_samples(millis):
    return sample_rate[0] * millis

def ideal_delay_by_5ms(samples = speech_samples):
    ideal_delays = []
    for i in range(0, len(samples)):
        input_signal = samples[i]
        output_signal = []
        n0 = sample_rate[i] * 0.005  # how many signals in 5 msecs?
        for n in range(0, len(input_signal)):
            if n < n0:
                output_signal.append(0)
            else:
                output_signal.append(input_signal[int(n-n0)])  # We can int() the indice as n-n0 always an integer
        ideal_delays.append(output_signal)

    return ideal_delays

def moving_average(k1, k2, input_signals = speech_samples):
    outputs = []
    for signal_index in range(0, len(input_signals)):
        input_signal = input_signals[signal_index]
        output_signal = []
        for i in range(0, k1):
            output_signal[i] = mean(input_signal[0, i+k2])  # Was 0, i
        for i in range(k1, len(input_signal) - k2):
            output_signal[i] = mean(input_signal[i-k1, i+k2])  # Should this be [i-k1, i]?
        for i in range(len(input_signal)-k2, len(input_signal) + 1):  # Was originally len(input_signal)-k2, k2
            output_signal[i] = mean(input_signal[i-k2, len(input_signal)])
        outputs.append(output_signal)
    return outputs

# Another couple of helper functions
def sign(n):
    return 1 if n >= 0 else 0

def sign_difference(a, b):
    return abs( (sign(a)-sign(b))/2 )

# A function that averages the convolution of a signal optionally transformed by some input function
def averaged_convolution(initial_input_signal, window = 30, modifier_function = lambda x: x):
    output_signal = []
    input_signal = map(modifier_function, initial_input_signal)
    window = time_to_samples(window)

    for i in range(0, window):
        output_signal.append(sum(map(lambda x: x/i, input_signal[0 : i])))

    input_signal = map(lambda x: x/window, input_signal)

    for i in range(window, len(input_signal)):
        output_signal[i] = sum(input_signal[i-window, i])

    return output_signal

def average_convolutions_of_signals(signals, modifier_function = lambda x: x, window = 30):
    output_signals = []
    for input_signal in signals:
        output_signals.append(averaged_convolution(input_signal, window, modifier_function))
    return output_signals

def short_term_energy_signals(signals):
    return average_convolutions_of_signals(signals, lambda x: x**2)

def magnitude_of_signals(signals):
    return average_convolutions_of_signals(signals, abs)

def sign_difference_of_signal(signal):
    sign_difference_signal = [0]
    for i in range(1, len(signal)):
        sign_difference_signal.append(sign_difference(signal[i], signal[i-1]))
    return sign_difference_signal

def sign_differences(signals):
    return [sign_difference_of_signal for signal in signals]

def zero_crossing_rates(signals):
    return average_convolutions_of_signals(sign_differences(signals))

def average_of_signals(signals):
    return map(mean, signals)

def log_of_signal_values(signal_values):
    return map(log, signal_values)

def log_of_average_of_signals(signals):
    return log_of_signal_values( average_of_signals(signals) )

# Each of these are arrays of the log of the average of the energy and magnitude and the average of the zero crossing rate for all of the signals, where the signal's index is the number o the signal in the input.
'''
e = log_of_average_of_signals( short_term_energy_signals(speech_samples) )
m = log_of_average_of_signals( magnitude_of_signals(speech_samples) )
z = average_of_signals( zero_crossing_rates(speech_samples) )
'''
