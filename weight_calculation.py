import bpy
import numpy as np
import tensorflow as tf

def calculate_weights_using_tensorflow_predictions(test_model, predictions):
    v_weights = []
    v_weights_indices = []
    for v in range(0, len(test_model.labels) ):
        weights_of_current_v = []
        indices_for_weights_of_current_v = []
        sum_of_weights = 0
        for bone in test_model.labels[v]:
            sum_of_weights += predictions[v][bone]
            weights_of_current_v.append(predictions[v][bone])
            indices_for_weights_of_current_v.append(bone)
        for i in range(0, len(weights_of_current_v) ):
            if sum_of_weights > 0:
                weights_of_current_v[i] = weights_of_current_v[i] / sum_of_weights
        v_weights.append(weights_of_current_v)
        v_weights_indices.append(indices_for_weights_of_current_v)
    return v_weights, v_weights_indices
