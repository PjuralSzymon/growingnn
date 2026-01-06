"""
Activation functions for neural networks.
"""
import numpy as np
from ..config import config


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
        def exe(X):
            """Execute ReLU activation."""
            return np.maximum(X, 0)
        
        @staticmethod
        def der(X):
            """Calculate ReLU derivative."""
            return np.where(X > 0, 1, 0)
        
    class leaky_ReLu:
        """Leaky Rectified Linear Unit activation function."""
        __name__ = 'leaky_ReLu'

        @staticmethod
        def exe(X):
            """Execute Leaky ReLU activation."""
            return np.where(X > 0, X, X * 0.001)
        
        @staticmethod
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
            if config.ENABLE_CLIP_ON_ACTIVATIONS:
                from ..helpers import clip
                return clip(result, 0.0001, 0.999)
            else:
                return result
        
        @staticmethod
        def der(X):
            """Calculate SoftMax derivative."""
            return 1.0
        
    class Sigmoid:
        """Sigmoid activation function."""
        __name__ = 'Sigmoid'

        @staticmethod
        def exe(X):
            """Execute Sigmoid activation."""
            return 1/(1 + np.exp(-X))
        
        @staticmethod
        def der(X):
            """Calculate Sigmoid derivative."""
            sigm = 1/(1 + np.exp(-X))
            return sigm * (1.0 - sigm)
        
    class Tanh:
        """Hyperbolic Tangent activation function."""
        __name__ = 'Tanh'

        @staticmethod
        def exe(X):
            """Execute Tanh activation."""
            return np.tanh(X)
        
        @staticmethod
        def der(X):
            """Calculate Tanh derivative."""
            return 1 - np.tanh(X)**2
    
    class Linear:
        """Linear (identity) activation function."""
        __name__ = 'Linear'
        
        @staticmethod
        def exe(X):
            """Execute Linear activation."""
            return X
        
        @staticmethod
        def der(X):
            """Calculate Linear derivative."""
            return np.ones_like(X)
