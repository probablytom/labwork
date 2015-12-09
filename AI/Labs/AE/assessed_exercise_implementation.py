from assessed_exercise_methods import *
import numpy as np
import matplotlib.pyplot as plt

import_input_data()
print "Calculating short term energy signals"
e = log_of_average_of_signals( short_term_energy_signals(speech_samples) )
print "Calculating magnitude of signals"
m = log_of_average_of_signals( magnitude_of_signals(speech_samples) )
print "Calculating zero crossing rates"
z = average_of_signals( zero_crossing_rates(speech_samples) )

'''
plt.scatter(e, m)
plt.show()
plt.scatter(e, z)
plt.show()
'''
plt.scatter(z, m)
plt.show()
