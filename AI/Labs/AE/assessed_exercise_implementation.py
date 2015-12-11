from assessed_exercise_methods import *
import numpy as np
import matplotlib.pyplot as plt
import assessed_exercise_methods

import_input_data()
correctness = 100
accuracy = 5
#set_training_type("gaussian")
set_training_type("euclidean")

feature_sets_speech = feature_sets_of(assessed_exercise_methods.speech_samples)
feature_sets_silence = feature_sets_of(assessed_exercise_methods.silence_samples)

for i in range(0, 50, accuracy):
    assessed_exercise_methods.training_sets_speech = feature_sets_speech[:i] + feature_sets_speech[i+accuracy:]
    assessed_exercise_methods.training_sets_silence = feature_sets_silence[:i] + feature_sets_silence[i+accuracy:]
    train()
    for j in range(0, accuracy):
        if not is_speech(feature_sets_speech[i + j]): correctness -= 1; print "Incorrect speech for " + str(i + j)
        if not is_silence(feature_sets_silence[i + j]): correctness -= 1; print "Incorrect silence for " + str(i + j)

print str(correctness) + "%"
