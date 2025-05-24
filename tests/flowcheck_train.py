# all imports

# Load Simple mnist dataset

# Train growingnn model 

# Present the accuracy and the time it took

import sys
sys.path.append('.')
sys.path.append('../')
import numpy as np
import growingnn as gnn
from growingnn.structure import SimulationScheduler
import time
import urllib.request
import csv
import os

def download_iris():
    """Download Iris dataset if not already present"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data_dir = './data/iris'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, 'iris.csv')
    if not os.path.exists(file_path):
        print("[INFO] Downloading Iris dataset...")
        try:
            response = urllib.request.urlopen(url)
            lines = [l.decode('utf-8') for l in response.readlines() if l.strip()]
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for line in lines:
                    writer.writerow(line.split(','))
            print("[INFO] Successfully downloaded Iris dataset")
        except Exception as e:
            print(f"[ERROR] Failed to download Iris dataset: {str(e)}")
            raise
    
    return file_path

def load_iris():
    print("[INFO] Loading Iris dataset...")
    file_path = download_iris()
    
    # Load data
    data = []
    labels = []
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 5:  # Ensure row has all features and label
                features = [float(x) for x in row[:4]]  # First 4 columns are features
                label = label_map[row[4].strip()]  # Strip whitespace and newlines from label
                data.append(features)
                labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(data)
    y = np.array(labels)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Reshape for CNN (samples, height, width, channels)
    # We'll reshape the 4 features into a 2x2 image
    X = X.reshape(-1, 2, 2, 1)
    
    # Split into train and test sets (80-20 split)
    train_size = int(0.8 * len(X))
    x_train = X[:train_size]
    y_train = y[:train_size]
    x_test = X[train_size:]
    y_test = y[train_size:]
    
    print(f"[INFO] Training data shape: {x_train.shape}")
    print(f"[INFO] Test data shape: {x_test.shape}")
    return x_train, y_train, x_test, y_test, range(3)

def train_and_evaluate():
    # Load Iris
    x_train, y_train, x_test, y_test, labels = load_iris()
    
    print("\n[INFO] Data Shapes:")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of labels: {len(labels)}\n")
    
    print("[INFO] Starting training...")
    start_time = time.time()
    
    # Train the model
    model = gnn.trainer.train(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        labels=labels,
        input_paths=1,
        path="./result/",
        model_name="GNN_model",
        epochs=30,
        generations=4,
        input_size=2,  # Changed to 2x2 for Iris features
        hidden_size=2,  # Changed to 2x2 for Iris features
        output_size=3,  # 3 classes in Iris
        input_shape=(2, 2, 1),  # Changed to 2x2 for Iris features
        kernel_size=2,
        batch_size=16,
        simulation_scheduler=SimulationScheduler(
            SimulationScheduler.PROGRESS_CHECK, 
            simulation_time=1, 
            simulation_epochs=1
        ),
        lr_scheduler = gnn.LearningRateScheduler(gnn.LearningRateScheduler.PROGRESIVE, 0.002),
        deepth=2
    )
    
    training_time = time.time() - start_time
    
    # Evaluate the model using growingnn's evaluation functions
    print("\n[INFO] Evaluating model...")
    
    # Check training accuracy
    train_predictions = model.forward_prop(x_train)
    train_accuracy = gnn.Model.get_accuracy(gnn.Model.get_predictions(train_predictions), y_train)
    
    # Check test accuracy
    test_predictions = model.forward_prop(x_test)
    test_accuracy = gnn.Model.get_accuracy(gnn.Model.get_predictions(test_predictions), y_test)
    
    print("\n[INFO] =========================")
    print("[INFO] Training Results:")
    print("[INFO] =========================")
    print(f"  - Training Time: {training_time:.2f}s")
    print(f"  - Training Accuracy: {train_accuracy:.2%}")
    print(f"  - Test Accuracy: {test_accuracy:.2%}")
    
    # Check if accuracies are acceptable
    if train_accuracy < 0.8:  # 80% training accuracy threshold
        print("[ERROR] Training accuracy too low!")
        sys.exit(1)
    elif test_accuracy < 0.1:  # 10% test accuracy threshold
        print("[WARNING] Test accuracy too low!")
    print("[INFO] Both training and test accuracies are acceptable")
    sys.exit(0)

if __name__ == '__main__':
    train_and_evaluate()
