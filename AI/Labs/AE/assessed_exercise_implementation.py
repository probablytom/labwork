from assessed_exercise_methods import *
import numpy as np
import matplotlib.pyplot as plt
import assessed_exercise_methods

def test():
    global correctness
    global accuracy
    correctness = 100
    for i in range(0, 50, accuracy):
        assessed_exercise_methods.training_sets_speech = feature_sets_speech[:i] + feature_sets_speech[i+accuracy:]
        assessed_exercise_methods.training_sets_silence = feature_sets_silence[:i] + feature_sets_silence[i+accuracy:]
        train()
        for j in range(0, accuracy):
            if not is_speech(feature_sets_speech[i + j]): correctness -= 1; print "Incorrect speech #" + str(i + j)
            if not is_silence(feature_sets_silence[i + j]): correctness -= 1; print "Incorrect silence #" + str(i + j)

import_input_data()
feature_sets_speech = feature_sets_of(assessed_exercise_methods.speech_samples)
feature_sets_silence = feature_sets_of(assessed_exercise_methods.silence_samples)
correctness = 100  # %
accuracy = 5

set_training_type("gaussian")
test()
print "Gaussian correctness: " + str(correctness) + "%"


set_training_type("euclidean")
test()
print "Euclidean correctness: " + str(correctness) + "%"
