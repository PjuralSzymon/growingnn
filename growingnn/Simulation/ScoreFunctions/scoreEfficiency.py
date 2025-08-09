import time
from ...config import Config
from ...structure import LearningRateScheduler, Conv, Layer

def scoreTime(M, epochs, X_train, Y_train):
    # more time smaller score
    start_time = time.time()
    _, _ = M.gradient_descent(X_train, Y_train, 2, LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.1) , True)
    end_time = time.time()
    time_difference_in_seconds = end_time - start_time
    grade = 1.0/(Config.TIME_EFFICIENCY_WEIGHT * time_difference_in_seconds + 1.0)
    return grade

def scoreCountWeights(M, epochs, X_train, Y_train):
    # more weights smaller score
    counter = 0
    for layer in M.hidden_layers + M.input_layers + [M.output_layer]:
        if type(layer) == Conv:
            counter += int(layer.depth) * int(layer.input_depth) * int(layer.kernel_size) * int(layer.kernel_size)
        elif type(layer) == Layer:
            counter += layer.input_size * layer.neurons
    # Use configurable weight instead of hardcoded 0.001
    grade = 1.0/(float(counter) * Config.WEIGHT_COUNT_WEIGHT + 1.0)
    return grade