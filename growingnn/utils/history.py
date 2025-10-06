"""
Training history tracking for neural networks.
"""
import json
import numpy as np
from numba import jit
from ..config import config
from ..helpers import get_numpy_array, get_list_as_numpy_array, NumpyArrayEncoder


class History:
    """Training history tracker."""
    
    def __init__(self, keys):
        """
        Initialize training history.
        
        Args:
            keys: List of keys to track in history
        """
        self.Y = {}
        self.last_img_id = 0
        self.description = "\ndescription_of_training_process: \n"
        self.best_train_acc = 0.0
        self.best_test_acc = 0.0
        for key in keys:
            self.Y[key] = []

    @staticmethod
    @jit(nopython=True)
    def _calculate_accuracy(correct_predictions, total_samples):
        """Calculate accuracy using Numba JIT."""
        return correct_predictions / total_samples

    def update_training_progress(self, correct_predictions, total_samples, total_loss, epoch, current_alpha, quiet):
        """Update training history and print progress."""
        acc = self._calculate_accuracy(correct_predictions, total_samples)
        self.append('accuracy', acc)
        self.append('loss', total_loss)
        return acc

    def get_length(self):
        """Get the length of history data."""
        return len(self.Y[list(self.Y.keys())[0]])

    def append(self, key, value):
        """Append a value to a specific key in history."""
        if not key in self.Y.keys():
            self.Y[key] = []
        self.Y[key].append(value)

    def merge(self, new_hist):
        """Merge another history object into this one."""
        for key in self.Y.keys():
            if key in self.Y.keys() and key in new_hist.Y.keys():
                self.Y[key] += new_hist.Y[key]

    def _check_learning_capability(accuracies, patience=15, verbose=0.3):
        """
        Determine if a model is still learning based on recent accuracy values.
        Returns True if the model is still learning (significant upward trend), 
        or False if learning has plateaued or stalled.
        """
        import numpy as np
        
        # If not enough data points, assume the model can still learn
        if len(accuracies) < 2:
            return True
        
        # Consider only the most recent `patience` accuracy values
        recent_values = accuracies[-patience:] if patience > 0 else accuracies
        if len(recent_values) < 2:
            # Not enough values in the window to determine a trend
            return True
        
        # Calculate the overall slope as the difference between the last and first accuracy in the window
        net_improvement = recent_values[-1] - recent_values[0]
        # Calculate the standard deviation of accuracies in the window to gauge fluctuations
        std_dev = float(np.std(recent_values))
        
        # Define a small improvement threshold based on the verbose (sensitivity) parameter
        # Lower verbose => more sensitive (higher threshold required), higher verbose => less sensitive (lower threshold)
        if verbose <= 0:
            verbose = 1  # avoid division by zero, treat non-positive verbose as most sensitive
        threshold = 0.01 / verbose  # e.g., verbose=1 -> 0.01 (1% accuracy), verbose=2 -> 0.005, verbose=10 -> 0.001
        
        # Check conditions for learning capability
        # Condition 1: If net improvement is below the threshold (no significant upward trend)
        if net_improvement <= threshold:
            return False  # No meaningful overall improvement (plateau or very slow progress)
        # Condition 2: If accuracy fluctuates a lot without a clear upward trend (improvement overshadowed by noise)
        if net_improvement > 0 and std_dev > net_improvement:
            return False  # Variations are larger than the overall improvement (no clear upward trend)
        
        # If neither condition triggered, the model is likely still learning (trend is upward)
        return True

    def learning_capable(self, epochsInGeneration=50):
        """Check if the model is still capable of learning based on accuracy history."""
        if 'accuracy' not in self.Y or len(self.Y['accuracy']) == 0:
            print("No accuracy data available")
            return True
            
        # Use all available data points up to patience
        recent_accuracies = np.array(self.Y['accuracy'][-min(len(self.Y['accuracy']), epochsInGeneration):])
        return History._check_learning_capability(recent_accuracies)

    def get_last(self, key):
        """Get the last value for a specific key."""
        return self.Y[key][-1]

    def draw_hist(self, label, path):
        """Draw and save history plots."""
        if not config.SAVE_PLOTS: 
            return
        import matplotlib.pyplot as plt
        
        for key in self.Y.keys():
            xc = range(0, len(self.Y[key]))
            plt.figure()
            plt.plot(xc, get_list_as_numpy_array(self.Y[key]), label=key)
            plt.legend()
            try:
                plt.savefig(os.path.abspath(path + "/" + label + "_" + key + ".png"))
            except Exception as e:
                print(f"Error saving plot: {e}")
            plt.close()
            
    def save(self, path):
        """Save history to file."""
        dict = {}
        dict['keys'] = {}
        for key in self.Y.keys():
            dict['keys'][key] = np.array(get_list_as_numpy_array(self.Y[key]))
        dict['last_img_id'] = str(self.last_img_id)
        dict['description'] = str(self.description)
        dict['best_train_acc'] = str(self.best_train_acc)
        dict['best_test_acc'] = str(self.best_test_acc)
        with open(path, 'w+') as f:
            f.write(json.dumps(dict, cls=NumpyArrayEncoder))
        with open(path+"_description.txt", 'w') as f:
            f.write(self.description)
            
    def load(self, path):
        """Load history from file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.Y = {}
        self.last_img_id = int(data['last_img_id'])
        self.description = str(data['description'])
        if 'best_train_acc' in data.keys():
            self.best_train_acc = float(data['best_train_acc'])
            self.best_test_acc = float(data['best_test_acc'])
        for key in data['keys'].keys():
            self.Y[key] = list(np.asarray(data['keys'][key]))
