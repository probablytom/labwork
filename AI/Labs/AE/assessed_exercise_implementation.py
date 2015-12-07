from assessed_exercise_methods import *

import_input_data()
e = log_of_average_of_signals( short_term_energy_signals(speech_samples) )
m = log_of_average_of_signals( magnitude_of_signals(speech_samples) )
z = average_of_signals( zero_crossing_rates(speech_samples) )


print e
print
print m
print
print z
