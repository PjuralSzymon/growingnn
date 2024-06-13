#import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import json
import random
import numpy
import cupy

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def switch_to_gpu():
    #print(" helper: switch_to_gpu")
    global np, IS_CUPY
    import cupy as np
    IS_CUPY = True

def switch_to_cpu():
    #print(" helper: switch_to_cpu")
    global np, IS_CUPY
    import numpy as np
    IS_CUPY = False
    
def clip(X, min, max):
    return np.array(np.clip(get_numpy_array(X), min, max))

def argmax(X, axis):
    return np.array(numpy.argmax(get_numpy_array(X), axis))

def randn(shape):
        return np.array(numpy.random.randn(shape))

def get_list_as_numpy_array(X):
    for i in range(0, len(X)):
        X[i] = get_numpy_array(X[i])
    return X
    
def get_numpy_array(X):
    if IS_CUPY == True:
        if isinstance(X, cupy.ndarray):
            return X.get()
        else:
            return numpy.array(X)
    else:
        return numpy.array(X)
    
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
    sum = array[0]
    for i in range(1,len(array)): 
        sum += array[i]
    return sum

def mean_n(array):
    sum = add_n(array)
    div = float(len(array))
    return sum / div

def mean_n_conv(array, shape):
    sum = array[0]
    for i in range(1,len(array)): 
        sum += array[i]
    div = float(len(array))
    return sum / div

def delete_repetitions(array):
    result = []
    for obj in array:
        if not obj in result:
            result.append(obj)
    return result

def eye_stretch(a,b):
    A = np.eye(max(a,b))
    return np.array(cv.resize(get_numpy_array(A), (a,b))).T

def strech(x, shape):
    result = np.zeros((shape[0], shape[1], x.shape[2]))
    for i in range(0, x.shape[2]):
        result[:,:,i] = np.array(cv.resize(get_numpy_array(x[:,:,i]), shape))
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
