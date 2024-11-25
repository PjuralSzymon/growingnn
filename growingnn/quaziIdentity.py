from matplotlib import pyplot as plt
from numba import jit
import numpy as np
import cv2 as cv
from .helpers import *

RESHEPERS = {}

def eye_stretch(a,b):
    A = np.eye(max(a,b))
    return np.ascontiguousarray(np.array(cv.resize(get_numpy_array(A), (a,b))).T)

def get_reshsper(size_from, size_to):
    if size_from == size_to:
        return None
    elif not (size_from, size_to) in RESHEPERS.keys():
        RESHEPERS[(size_from, size_to)] = eye_stretch(size_from, size_to)
    return RESHEPERS[(size_from, size_to)]

def Reshape(x, output_size, QIdentity):
    x_reshaped = np.zeros((output_size, x.shape[1]))
    for i in range(0, x.shape[1]):
        if QIdentity is None:
            x_reshaped[:, i] = x[:, i]
        else:
            x_reshaped[:, i] = np.dot(x[:, i], QIdentity)
    return x_reshaped

def Reshape_forward_prop(x, output_size, QIdentity):
    x_reshaped = np.zeros((output_size, x.shape[0]))
    for i in range(0, x.shape[0]):
        flatten_size = x[i].shape[0] * x[i].shape[1] * x[i].shape[2]
        flatten = np.reshape(x[i], flatten_size)
        if QIdentity is None:
            x_reshaped[:, i] = flatten
        else:
            x_reshaped[:, i] = flatten.dot(QIdentity)
    return x_reshaped

def Reshape_back_prop(E, input_shape, QIdentity):
    E_reshaped = np.zeros((E.shape[1], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(0, E.shape[1]):
        if QIdentity is None:
            x_reshaped = E[:, i]
        else:
            x_reshaped = E[:, i].dot(QIdentity)
        E_reshaped[i, :, :, :] = np.reshape(x_reshaped, input_shape)
    return E_reshaped
