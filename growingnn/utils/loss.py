"""
Loss functions for neural networks.
"""
import numpy as np
from numba import jit


class Loss:
    """Container class for loss functions."""
    
    @staticmethod
    def getByName(name):
        """Get loss function by name."""
        if name == Loss.MSE.__name__:
            return Loss.MSE
        elif name == Loss.MAE.__name__:
            return Loss.MAE
        elif name == Loss.multiclass_cross_entropy.__name__:
            return Loss.multiclass_cross_entropy

    class MSE:
        """Mean Squared Error loss function."""
        __name__ = 'MSE'
        
        @staticmethod
        def exe(Y_true, Y_pred):
            """Execute MSE loss calculation."""
            return np.sum((Y_pred - Y_true)**2)/Y_pred.shape[0]
        
        @staticmethod
        def der(Y_true, Y_pred):
            """Calculate MSE derivative."""
            return Y_pred - Y_true
    
    class MAE:
        """Mean Absolute Error loss function."""
        __name__ = 'MAE'
        
        @staticmethod
        def exe(Y_true, Y_pred):
            """Execute MAE loss calculation."""
            return np.sum(np.abs(Y_pred - Y_true))/Y_pred.shape[0]
        
        @staticmethod
        def der(Y_true, Y_pred):
            """Calculate MAE derivative."""
            return np.sign(Y_pred - Y_true)
    
    class multiclass_cross_entropy:
        """Multiclass Cross Entropy loss function.
        
        Note: When used with softmax activation, the combined gradient
        of softmax + cross-entropy simplifies to (Y_pred - Y_true).
        This avoids computing the expensive softmax Jacobian.
        """
        __name__ = 'multiclass_cross_entropy'
        
        @staticmethod
        def exe(Y_true, Y_pred):
            """Execute multiclass cross entropy loss calculation."""
            error = 0.0
            for i in range(0, Y_true.shape[1]):
                error -= np.dot(Y_true[:,i].T, np.log(Y_pred[:,i]))
            return error / Y_true.shape[1]
        
        @staticmethod
        def der(Y_true, Y_pred):
            """Calculate multiclass cross entropy derivative."""
            grad = np.zeros(Y_true.shape)
            for i in range(0, Y_true.shape[1]):
                partial_grad = -Y_true[:, i] / Y_pred[:, i]
                A = np.tile(np.reshape(Y_pred[:, i], (Y_pred.shape[0], 1)), (1, Y_pred.shape[0]))
                grad[:, i] = (A * (np.identity(Y_pred.shape[0]) - A.T)) @ partial_grad
            return grad