import time
from ...structure import *

def scoreTime(M, epochs, X_train, Y_train):
    # more time smaller score
    start_time = time.time()
    _, _ = M.gradient_descent(X_train, Y_train, 1, LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.1) , True)
    end_time = time.time()
    time_difference_in_seconds = end_time - start_time
    return (1/(time_difference_in_seconds + 1))


def scoreCountWeights(M, epochs, X_train, Y_train):
    # more time smaller score
    counter = 0
    for layer_id in M.hidden_layers:
        layer = M.get_layer(layer_id)
        if type(layer) == Conv:
            counter += int(layer.depth) * int(layer.input_depth) * int(layer.kernel_size) * int(layer.kernel_size)
        elif type(layer) == Layer:
            counter += layer.input_size * layer.neurons
    return (1/(counter + 1))
