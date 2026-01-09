import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numba import jit
import cv2 as cv
import json
import random
import numpy as np

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def get_reverse_normal_distribution(clip_range, shape):
    # Calculate total number of samples needed
    num_samples = np.prod(shape)
    
    # Generate samples for left and right sides directly
    left_samples = np.random.normal(loc=-2.2 * clip_range, scale=clip_range, size=num_samples)
    right_samples = np.random.normal(loc=2.2 * clip_range, scale=clip_range, size=num_samples)
    
    # Merge and shuffle samples
    merged = np.concatenate((left_samples, right_samples))
    np.random.shuffle(merged)
    
    # Take the first num_samples elements and reshape
    result = merged[:num_samples].reshape(shape)
    
    # Ensure all values are within the expected range
    result = np.clip(result, -2.2 * clip_range, 2.2 * clip_range)
    
    return result

# GPU/CuPy functionality removed - using CPU only
    
def clip(X, min_val, max_val):
    """Clip array values to range. Optimized to avoid unnecessary array copies."""
    if isinstance(X, np.ndarray):
        # Already numpy - use in-place if possible, otherwise direct clip
        return np.clip(X, min_val, max_val)
    # Convert to numpy only if needed
    return np.clip(np.asarray(X), min_val, max_val)

def argmax(X, axis):
    return np.array(np.argmax(get_numpy_array(X), axis))

def randn(shape):
        return np.array(np.random.randn(shape))

def get_list_as_numpy_array(X):
    for i in range(0, len(X)):
        X[i] = get_numpy_array(X[i])
    return X
    
def get_numpy_array(X):
    # CPU only - no GPU conversion needed
    return np.array(X)
    
def convert_to_desired_type(X):
    if not isinstance(X, np.ndarray):
        return np.array(X)
    return X
    
def one_hot(Y, Y_max = 0):
    Y_max = int(max(Y_max, Y.max() + 1))
    one_hot_Y = np.zeros((Y.size, Y_max))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def add_n(array):
    """Sum all arrays in a list element-wise. Optimized using np.stack."""
    if len(array) == 1:
        return array[0]
    return np.sum(np.stack(array, axis=0), axis=0)

def mean_n(array):
    """Compute element-wise mean of arrays in a list. Optimized using np.stack + np.mean."""
    if len(array) == 1:
        return array[0]
    # All arrays have same shape after Reshape() in back_prop, so np.stack works
    return np.mean(np.stack(array, axis=0), axis=0)

def mean_n_conv(array, shape):
    """Compute element-wise mean of conv arrays. Optimized using np.stack + np.mean."""
    if len(array) == 1:
        return array[0]
    # All arrays have same shape after Resize() in back_prop, so np.stack works
    return np.mean(np.stack(array, axis=0), axis=0)

def delete_repetitions(array):
    result = []
    for obj in array:
        if not obj in result:
            result.append(obj)
    return result



def strech(x, shape):
    x_np = get_numpy_array(x) if not isinstance(x, np.ndarray) else x
    result = np.empty((shape[0], shape[1], x_np.shape[2]))
    # Process each channel
    for i in range(0, x_np.shape[2]):
        result[:,:,i] = cv.resize(x_np[:,:,i], shape)
    return result

def draw_hist(hist, label, path):
    xc = range(0, len(hist['train']))
    plt.figure()
    plt.plot(xc, hist['train'], label = "train acc")
    plt.plot(xc, hist['test'], label = "test acc")
    plt.legend()
    plt.savefig(path + "/" +label +".png")
    plt.close()

def get_max_loss(y):
    examples = 1
    for s in y.shape: examples *= s
    return examples * np.max(y)

def train_test_split_many_inputs(x, y, test_size):
    x = np.swapaxes(x, 0, 1)
    x_train = x[int(x.shape[0]* test_size) : x.shape[0]]
    x_test = x[0: int(x.shape[0] * test_size)]
    y_train = y[int(y.shape[0]* test_size) : y.shape[0]]
    y_test = y[0: int(y.shape[0] * test_size)] 

    x_train = np.swapaxes(x_train, 0, 1)
    x_test = np.swapaxes(x_test, 0, 1)
    return x_train, x_test, y_train, y_test

def protected_sampling(x, y, n):
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    samples_per_class = max(1, n // num_classes)
    selected_indices = []
    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]
        selected_indices.extend(np.random.choice(class_indices, size=min(samples_per_class, len(class_indices))))
    return select_data_at_indices(x, y, selected_indices)

def select_data_at_indices(x, y, selected_indices):
    matching_dim_x = None
    matching_dim_y = None # should be allways 0
    shape_x = list(x.shape)
    shape_y = list(y.shape)
    for i in range(0, len(shape_x)):
        for j in range(0, len(shape_y)):
            if shape_x[i] == shape_y[j]:
                matching_dim_x = i
                matching_dim_y = j

    if matching_dim_x is None:
        raise ValueError("Nie znaleziono odpowiedniego wymiaru zgodnego dla x i y.")
    x_selected = np.take(x, selected_indices, axis=matching_dim_x)
    y_selected = np.take(y, selected_indices, axis=matching_dim_y)

    return x_selected, y_selected

def limit_classes(x_train, y_train, x_test, y_test, num_classes=5):
    selected_indices_train = np.where(y_train < num_classes)
    selected_indices_test = np.where(y_test < num_classes)
    x_train = x_train[selected_indices_train]
    y_train = y_train[selected_indices_train]
    x_test = x_test[selected_indices_test]
    y_test = y_test[selected_indices_test]
    return x_train, y_train, x_test, y_test

def set_seed(new_seed):
    np.random.seed(new_seed)
    random.seed(new_seed)
