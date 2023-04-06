import numpy as np
import tensorflow as tf
import math

def decode_vector(vector):
    return [i for i, x in enumerate(vector) if x == 1]

def get_top_n_indices(vector, n):
    return np.argpartition(vector, -n)[-n:]

def top_rated_percent(y_true, y_pred):
    accuracies = []

    for i in range(len(y_true)):
        true_cube = decode_vector(y_true[i])
        pred_cube = get_top_n_indices(y_pred[i], math.ceil(len(true_cube) * 1.2))
        
        accuracy = (len(set(true_cube) & set(pred_cube)) + 1) / (len(true_cube) + 1)
        accuracies.append(accuracy)

    return np.mean(accuracies)

def relative_pick(x, y_true, y_pred):
    masked = x[0] * y_pred

    accuracies = []

    for i in range(len(y_true)):
        difference = y_true[i] - masked[i]
        accuracies.append(np.mean(np.abs(difference)))

    return np.mean(accuracies)
