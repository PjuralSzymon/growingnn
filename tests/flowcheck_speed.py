import sys
sys.path.append('.')
sys.path.append('../')
import unittest
import numpy as np
import growingnn as gnn
from growingnn.structure import SimulationScheduler
from testSuite import mode
import time


def get_dataset(datasize, datadimensionality, classes):
    x_conv_train = np.random.random((datasize, datadimensionality, datadimensionality, 1))
    y_conv_train = np.random.randint(classes, size=(datasize, ))
    x_conv_test = np.random.random((int(datasize / 2), datadimensionality, datadimensionality, 1))
    y_conv_test = np.random.randint(classes, size=(int(datasize / 2), ))
    labels = range(0, classes)

    return x_conv_train, y_conv_train, x_conv_test, y_conv_test, labels

def timer_train(datasize, datadimensionality, classes):
    print("[INFO] Starting to generate the dataet")
    x_conv_train, y_conv_train, x_conv_test, y_conv_test, labels = get_dataset(datasize, datadimensionality, classes)
    print("[INFO] Dataset generated")
    start_time = time.time()
    gnn.trainer.train(
        x_train=x_conv_train,
        y_train=y_conv_train,
        x_test=x_conv_test,
        y_test=y_conv_test,
        labels=labels,
        input_paths=1,
        path="./result",
        model_name="GNN_model",
        epochs=1,
        generations=1,
        input_size=datadimensionality,
        hidden_size=datadimensionality,
        output_size=classes,
        input_shape=(datadimensionality, datadimensionality, 1),
        kernel_size=2,
        batch_size=1,
        simulation_scheduler = SimulationScheduler(SimulationScheduler.PROGRESS_CHECK, simulation_time = 2, simulation_epochs = 2), 
        deepth=2
        )
    end_time = time.time()
    return round(end_time - start_time, 2)

if __name__ == '__main__':
    TIMING_RESULTS = {"easy": "...", "mid": "...", "hard": "..."}

    print("[INFO] Starting timing for basic dataset")
    train_time = timer_train(10, 5, 5) # basic settings
    print("[INFO] Timing finished, result: ", TIMING_RESULTS)

    print("[INFO] Starting timing for easy dataset")
    train_time = timer_train(500, 28, 10) # MNIST settings
    TIMING_RESULTS["easy"] = train_time
    print("[INFO] Timing finished, result: ", TIMING_RESULTS)

    print("[INFO] Starting timing for mid dataset")
    train_time = timer_train(1000, 30, 10) # Something in beetween
    TIMING_RESULTS["mid"] = train_time
    print("[INFO] Timing finished, result: ", TIMING_RESULTS)

    print("[INFO] Starting timing for hard dataset")
    train_time = timer_train(6000, 32, 10) # CIFAR settings (600 instead of 6000)
    TIMING_RESULTS["hard"] = train_time
    print("[INFO] =========================")
    print("[INFO] Timing finished, result: ", TIMING_RESULTS)
    print("[INFO] =========================")

    if TIMING_RESULTS["easy"] < 6:
        print("[INFO] Time for basic run is acceptable")
        sys.exit(0)  # Exit with code 0 if all tests passed successfully
    elif TIMING_RESULTS["mid"] < 10:
        print("[INFO] Time for mid run is acceptable")
        sys.exit(0)  # Exit with code 0 if all tests passed successfully
    elif TIMING_RESULTS["hard"] < 50:
        print("[INFO] Time for hard run is acceptable")
        sys.exit(0)  # Exit with code 0 if all tests passed successfully
    else:
        print("[ERROR] Time for basic run took too much time")
        sys.exit(1)  # Exit with code 1 if there were failures or errors in tests
