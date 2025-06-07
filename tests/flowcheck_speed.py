import sys
sys.path.append('.')
sys.path.append('../')
import unittest
import numpy as np
import growingnn as gnn
from growingnn.structure import SimulationScheduler
from testSuite import mode
import time

EXPERIMENT_REPETITIONS = 5
BASE_TIME_LINE = {
    "easy": 1.26,
    "mid":  1.58,
    "hard": 9.17
}

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
    times = []
    for i in range(2):
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
        times.append(end_time - start_time)
    return float(round(np.mean(times), 2))

def calculate_performance_change(current_time, baseline_time):
    """Calculate percentage change in performance"""
    if current_time > baseline_time:
        return ((current_time - baseline_time) / baseline_time) * 100
    else:
        return ((baseline_time - current_time) / baseline_time) * -100

def print_performance_report(timing_results):
    print("\n[INFO] =========================")
    print("[INFO] Performance Report:")
    print("[INFO] =========================")
    
    total_current = 0
    total_baseline = 0
    
    for test_type, current_time in timing_results.items():
        baseline = BASE_TIME_LINE[test_type]
        total_current += current_time
        total_baseline += baseline
    avg_change = calculate_performance_change(total_current, total_baseline)
    print(f"\n[INFO] Overall Performance:")
    print(f"  - Total Current Time: {total_current:.2f}s")
    print(f"  - Total Baseline Time: {total_baseline:.2f}s")
    print(f"  - Average Change: {avg_change:+.2f}%")
    
    if abs(avg_change) < 1:
        print("  - Status: SAME PERFORMANCE")
    elif avg_change > 0:
        print(f"  - Status: SLOWER (by {avg_change:+.2f}%)")
    else:
        print(f"  - Status: FASTER (by {abs(avg_change):+.2f}%)")
    
    print("[INFO] =========================")

if __name__ == '__main__':
    TIMING_RESULTS = {"easy": "...", "mid": "...", "hard": "..."}

    times_easy = []
    times_mid = []
    times_hard = []
    for i in range(EXPERIMENT_REPETITIONS):
        print("[INFO] Starting timing for iteration ", i)
        train_time = timer_train(500, 28, 10) # MNIST settings
        times_easy.append(train_time)

        train_time = timer_train(1000, 30, 10) # Something in beetween
        times_mid.append(train_time)

        train_time = timer_train(6000, 32, 10) # CIFAR settings (600 instead of 6000)
        times_hard.append(train_time)

        TIMING_RESULTS["easy"] = float(round(np.mean(times_easy), 2))
        TIMING_RESULTS["mid"] = float(round(np.mean(times_mid), 2))
        TIMING_RESULTS["hard"] = float(round(np.mean(times_hard), 2))   
        print("[INFO] Timing finished, result: ", TIMING_RESULTS)
    print("[INFO] =========================")
    print("[INFO]               Base Line: ", BASE_TIME_LINE)
    print("[INFO] Timing finished, result: ", TIMING_RESULTS)
    print_performance_report(TIMING_RESULTS)
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
        print("[ERROR] Time for training takes too much time optimization errors")
        sys.exit(1)  # Exit with code 1 if there were failures or errors in tests
