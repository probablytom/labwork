from assessed_exercise_methods import *
import numpy as np
import matplotlib.pyplot as plt
import assessed_exercise_methods

import_input_data()
correctness = 100
for i in range(0, 50, 5):
    assessed_exercise_methods.training_sets_speech = feature_sets_of(speech_samples[:i] + feature_sets_of(speech_samples[i+5:]))
    assessed_exercise_methods.training_sets_silence = feature_sets_of(silence_samples[:i] + feature_sets_of(silence_samples[i+5:]))
    train()
    for j in range(0, 5):
        print "i: " + str(i), "\tj: " + str(j)  
        if not is_speech(speech_samples[i + j]): correctness -= 1
        if not is_silence(silence_samples[i + j]): correctness -= 1

print str(correctness) + "%"

'''
import_input_data()
print "Calculating short term energy signals"
e = log_of_average_of_signals( short_term_energy_signals(speech_samples) )
print "Calculating magnitude of signals"
m = log_of_average_of_signals( magnitude_of_signals(speech_samples) )
print "Calculating zero crossing rates"
z = average_of_signals( zero_crossing_rates(speech_samples) )
'''
'''
plt.scatter(e, m)
plt.show()
plt.scatter(e, z)
plt.show()
plt.scatter(z, m)
plt.show()
'''
