from numpy import mean, log
from math import pi, sqrt, exp

noise_samples = silence_samples, speech_samples = [], []
training_sets_speech = []
training_sets_silence = []
metrics = []
sample_rate = []
training_type = "gaussian"

def import_input_data():
    global speech_samples
    global silence_samples
    global sample_rate
    speech_samples = []
    silence_samples = []
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
        silence_samples.append(input_data)
    
    for i in range(1, len(speech_samples) + 1):
        sample_rate.append( int( len(speech_samples[i-1])/0.3 ) )  # Sample is always 0.3 seconds long. 
    for i in range(1, len(silence_samples) + 1):
        sample_rate.append( int( len(silence_samples[i-1])/0.3 ) )  # Sample is always 0.3 seconds long. 


# Just as a helper function:
def time_to_samples(millis):
    return int(sample_rate[0] * millis / 1000)
    

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
            output_signal[i] = mean(input_signal[i-k1, len(input_signal)])
        outputs.append(output_signal)
    return outputs

# A function that averages the convolution of a signal optionally transformed by some input function
def averaged_convolution(initial_input_signal, window = 30, modifier_function = lambda x: x):
    output_signal = []
    input_signal = map(modifier_function, initial_input_signal)
    window = time_to_samples(window)

    for i in range(0, window):
        output_signal.append(sum(map(lambda x: x/float(i), input_signal[0 : i])))

    input_signal = map(lambda x: x/window, input_signal)

    for i in range(window, len(input_signal)):
        output_signal.append(sum(input_signal[i-window : i]))

    return output_signal

def average_convolutions_of_signals(signals, modifier_function = lambda x: x, window = 30):
    return [averaged_convolution(signal, window, modifier_function) for signal in signals]

def short_term_energy_signals(signals):
    return average_convolutions_of_signals(signals, lambda x: x**2)

def magnitude_of_signals(signals):
    return average_convolutions_of_signals(signals, abs)

# Another couple of helper functions
def sign(n):
    return 1 if n >= 0 else 0

def sign_difference(a, b):
    return abs( (sign(a)-sign(b)) )

def sign_difference_of_signal(signal):
    sign_difference_signal = [0]
    for i in range(1, len(signal)):
        sign_difference_signal.append(sign_difference(signal[i], signal[i-1]))
    return sign_difference_signal

def sign_differences(signals):
    return [sign_difference_of_signal(signal) for signal in signals]

def zero_crossing_rates(signals):
    return average_convolutions_of_signals(sign_differences(signals), lambda x: x / float(2))

def average_of_signals(signals):
    return map(mean, signals)

def log_of_signal_values(signal_values):
    return map(log, signal_values)

def log_of_average_of_signals(signals):
    return log_of_signal_values( average_of_signals(signals) )

def feature_sets_of(signals, silence = False):
    e = log_of_average_of_signals( short_term_energy_signals(signals) )
    m = log_of_average_of_signals( magnitude_of_signals(signals) )
    z = average_of_signals( zero_crossing_rates(signals) )
    return zip(e, m, z)

def feature_set_of(signal):
    e = log_of_average_of_signals(short_term_energy_signals([signal]))[0]
    m = log_of_average_of_signals(magnitude_of_signals([signal]))[0]
    z = average_of_signals( zero_crossing_rates([signal]) )[0]
    return [e, m, z]

def construct_variance_modifier(mean):
    def variance_modifier(feature_value):
        return (feature_value - mean) # ** 2
    return variance_modifier

def gaussian_mean(training_set):
    training_transposed = zip(training_set[0], training_set[1], training_set[2])
    return [(sum(training_transposed[i]))/float(len(training_set)) for i in range(0, 3)]

def gaussian_variance(training_set):
    training_mean = gaussian_mean(training_set)
    training_transposed = zip(training_set[0], training_set[1], training_set[2])
    return [(sum(map(construct_variance_modifier(training_mean[i]), training_transposed[i] )))/float(len(training_set)) for i in range(0, 3)]

def sigma_phi(sigma):
    return 1/sqrt(2 * pi)/sigma

# Make a gaussian for a given mean and variance. 
def construct_gaussian(mean, variance):
    def gaussian(feature_value):
        return sigma_phi(variance) * exp(0-( ((feature_value - mean)**2) / (2*(variance)**2) ))
    return gaussian

def train_gaussian():
    global training_sets_speech
    global training_sets_silence
    global speech_samples
    global silence_samples
    global metrics
    
    speech_mean = gaussian_mean(training_sets_speech)
    speech_variance = gaussian_variance(training_sets_speech)
    silence_mean = gaussian_mean(training_sets_silence)
    silence_variance = gaussian_variance(training_sets_silence)
    
    metrics = []
    metrics.extend([construct_gaussian(speech_mean[i], speech_variance[i]) for i in range(0, 3)])
    metrics.extend([construct_gaussian(silence_mean[i], silence_variance[i]) for i in range(0, 3)])
    
    # Metrics now contains the gaussians for speech e, speech m, speech z, silence e, silence m, and silence z in that order. 

def spatial_distance(trained_set, set_to_test):
    return sqrt(sum([(trained_set[i] - set_to_test[i])**2 for i in range(0, len(trained_set))]))

def train_euclidean():
    global training_sets_speech
    global training_sets_silence
    global speech_samples
    global silence_samples
    global metrics
    
    metrics = []
    
    # What do?
    
def mean_spatial_distance_speech(test_feature_set):
    global training_sets_speech
    return mean( [ spatial_distance(training_set, test_feature_set) for training_set in training_sets_speech ] )

def mean_spatial_distance_silence(test_feature_set):
    global training_sets_silence
    return mean( [ spatial_distance(training_set, test_feature_set) for training_set in training_sets_silence ] )

def product(values):
    if len(values) == 1: return values[0]
    else:                return values[0] * product( values[1:] )

# Discriminant via sum of loglikelihoods
def speech_metric_product(signal):
    return sum([ log(metrics[i](signal[i])) for i in range(0, 3) ])

# Discriminant via sum of loglikelihoods
def silence_metric_product(signal):
    return sum([ log(metrics[i+3](signal[i])) for i in range(0, 3) ])

def train():
    global training_type
    if training_type == "gaussian": train_gaussian()
    else                          : train_euclidean()

def is_speech(signal):
    global training_type
    #signal = feature_set_of(signal)
    if training_type == "gaussian": return speech_metric_product(signal) > silence_metric_product(signal)
    else                          : return mean_spatial_distance_speech(signal) < mean_spatial_distance_silence(signal) # euclidian distance metric calculation

def is_silence(signal):
    return not is_speech(signal)

def set_training_type(training_type_to_set):
    global training_type
    training_type = training_type_to_set
