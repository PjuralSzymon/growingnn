import matplotlib

from .config import config
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numba import jit
import numpy as np
import cv2 as cv
from .helpers import *

RESHEPERS = {}

def eye_stretch(a, b):
    if a == b:
        return np.eye(a)
    A = np.eye(max(a, b))
    return cv.resize(A, (a, b)).T

def get_reshsper(size_from, size_to):
    if size_from == size_to:
        return None
    key = (size_from, size_to)
    if key not in RESHEPERS:
        RESHEPERS[key] = np.ascontiguousarray(eye_stretch(size_from, size_to), dtype=config.FLOAT_TYPE)
    return RESHEPERS[key]

def Reshape(x, output_size, QIdentity):
    if QIdentity is None:
        return x[:output_size, :]
    x_reshaped = np.empty((output_size, x.shape[1]))
    for i in range(x.shape[1]):
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
