from assessed_exercise_methods import *
import numpy as np
import matplotlib.pyplot as plt
import assessed_exercise_methods

import_input_data()

e_speech = log_of_average_of_signals( short_term_energy_signals(assessed_exercise_methods.speech_samples) )
m_speech = log_of_average_of_signals( magnitude_of_signals(assessed_exercise_methods.speech_samples) )
z_speech = average_of_signals( zero_crossing_rates(assessed_exercise_methods.speech_samples) )
e_silence = log_of_average_of_signals( short_term_energy_signals(assessed_exercise_methods.silence_samples) )
m_silence = log_of_average_of_signals( magnitude_of_signals(assessed_exercise_methods.silence_samples) )
z_silence = average_of_signals( zero_crossing_rates(assessed_exercise_methods.silence_samples) )

plt.scatter(e_speech, z_speech, color="red")
plt.scatter(e_silence, z_silence, color="blue")
plt.legend(["speech", "silence"], loc="upper right")
plt.xlabel("Energy")
plt.ylabel("ZCR")
plt.grid()
plt.title("ZCR vs Energy")
plt.show()


plt.scatter(m_speech, z_speech, color="red")
plt.scatter(m_silence, z_silence, color="blue")
plt.legend(["speech", "silence"], loc="upper right")
plt.xlabel("Magnitude")
plt.ylabel("ZCR")
plt.grid()
plt.title("ZCR vs Magnitude")
plt.show()


plt.scatter(e_speech, m_speech, color="red")
plt.scatter(e_silence, m_silence, color="blue")
plt.legend(["speech", "silence"], loc="upper right")
plt.xlabel("Energy")
plt.ylabel("Magnitude")
plt.grid()
plt.title("Energy vs Magnitude")
plt.show()
