"""
Activation functions for neural networks.
"""
import numpy as np
from numba import jit
from ..config import config
from ..helpers import clip

class Activations:
    """Container class for activation functions."""
    
    @staticmethod
    def getByName(name):
        """Get activation function by name."""
        if name == Activations.ReLu.__name__:
            return Activations.ReLu
        elif name == Activations.leaky_ReLu.__name__:
            return Activations.leaky_ReLu
        elif name == Activations.SoftMax.__name__:
            return Activations.SoftMax
        elif name == Activations.Sigmoid.__name__:
            return Activations.Sigmoid
        elif name == Activations.Tanh.__name__:
            return Activations.Tanh
        elif name == Activations.Linear.__name__:
            return Activations.Linear
    
    class ReLu:
        """Rectified Linear Unit activation function."""
        __name__ = 'ReLu'

        @staticmethod
        @jit(nopython=True)
        def exe(X):
            """Execute ReLU activation."""
            return np.maximum(X, 0)
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            """Calculate ReLU derivative."""
            return np.where(X > 0, 1, 0)
        
    class leaky_ReLu:
        """Leaky Rectified Linear Unit activation function."""
        __name__ = 'leaky_ReLu'

        @staticmethod
        @jit(nopython=True)
        def exe(X):
            """Execute Leaky ReLU activation."""
            return np.where(X > 0, X, X * 0.001)
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            """Calculate Leaky ReLU derivative."""
            return np.where(X > 0, 1, 0.001)
        
    class SoftMax:
        """SoftMax activation function."""
        __name__ = 'SoftMax'

        @staticmethod
        def exe(X):
            """Execute SoftMax activation."""
            # Vectorized implementation instead of loop
            exp_X = np.exp(X - np.max(X, axis=0))
            result = exp_X / np.sum(exp_X, axis=0)
            if np.isnan(result).any():
                return Activations.exe_nan_safe(X)
            if config.ENABLE_CLIP_ON_ACTIVATIONS:
                return clip(result, 0.00001, 0.9999)
            else:
                return result
        
        @staticmethod
        def exe_nan_safe(X):
            result = np.zeros(X.shape)
            for i in range(0, X.shape[1]):
                exp = np.exp(X[:, i] - np.nanmax(X[:, i]))
                result[:, i] = np.nan_to_num(exp / np.sum(exp))
            return clip(result, 0.0001, 0.999)

        @staticmethod
        @jit(nopython=True)
        def der(X):
            """Calculate SoftMax derivative."""
            return 1.0
        
    class Sigmoid:
        """Sigmoid activation function."""
        __name__ = 'Sigmoid'

        @staticmethod
        @jit(nopython=True)
        def exe(X):
            """Execute Sigmoid activation."""
            return 1/(1 + np.exp(-X))
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            """Calculate Sigmoid derivative."""
            sigm = 1/(1 + np.exp(-X))
            return sigm * (1.0 - sigm)
        
    class Tanh:
        """Hyperbolic Tangent activation function."""
        __name__ = 'Tanh'

        @staticmethod
        @jit(nopython=True)
        def exe(X):
            """Execute Tanh activation."""
            return np.tanh(X)
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            """Calculate Tanh derivative."""
            return 1 - np.tanh(X)**2
    
    class Linear:
        """Linear (identity) activation function."""
        __name__ = 'Linear'
        
        @staticmethod
        @jit(nopython=True)
        def exe(X):
            """Execute Linear activation."""
            return X
        
        @staticmethod
        @jit(nopython=True)
        def der(X):
            """Calculate Linear derivative."""
            return np.ones_like(X)
