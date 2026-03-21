"""
Early stopping utilities for neural network training.
"""
from abc import ABC, abstractmethod

class BaseStopper(ABC):
    """
    Abstract base class for all training stoppers.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def check(self, model, x_train, y_train, epoch=None):
        """
        Check if training should stop.
        
        Args:
            model: The neural network model
            x_train: Test input data
            y_train: Test target data
            epoch: Current epoch number (optional)
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        pass
    
    def reset(self):
        """Reset the stopper state."""
        pass


class EmptyStopper(BaseStopper):
    """
    Never stops training.
    """
    def __init__(self):
        super().__init__()
    
    def check(self, model, x_train, y_train, epoch=None, params=None):
        return False

class AccuracyStopper(BaseStopper):
    """
    Stops training when accuracy reaches a target threshold.
    """
    def __init__(self, target_accuracy=0.9, metric_name="accuracy"):
        super().__init__()
        self.target_accuracy = target_accuracy
        self.metric_name = metric_name
    
    def check(self, model, x_train, y_train, epoch=None, params=None):
        should_stop = False
        current_accuracy = None
        if params is not None:
            if 'accuracy' in params:
                current_accuracy = params['accuracy']
        if model is None or x_train is None or y_train is None:
            return False
        if current_accuracy is None:
            current_accuracy = model.evaluate(x_train, y_train)
        if current_accuracy >= self.target_accuracy:
            should_stop = True
            msg = f"Stopping: {self.metric_name} reached {current_accuracy:.4f} (target: {self.target_accuracy:.4f})"
            if epoch is not None:
                msg += f" at epoch {epoch}"
            print(msg)
        return should_stop

class ParameterCountStopper(BaseStopper):
    """
    Stops training when parameter count decreases by a specified percentage.
    """
    def __init__(self, decrease_threshold=0.5, metric_name="parameter_count"):
        super().__init__()
        self.decrease_threshold = decrease_threshold  # 0.5 = 50% decrease
        self.metric_name = metric_name
        self.initial_parameter_count = None
        self.previous_parameter_count = None
    
    def check(self, model, x_train, y_train, epoch=None, params=None):
        """
        Check if parameter count has decreased by the threshold percentage from initial count.
        
        Args:
            model: The neural network model
            x_train: Test input data (not used in this implementation)
            y_train: Test target data (not used in this implementation)
            epoch: Current epoch number (optional)
            
        Returns:
            bool: True if parameter count decreased by threshold from initial, False otherwise
        """
        if model is None:
            return False
        should_stop = False
        # Get current parameter count using model.get_parametr_count()
        current_param_count = model.get_parametr_count()
        
        # Initialize tracking variables
        if self.initial_parameter_count is None:
            self.initial_parameter_count = current_param_count
            return should_stop
        
        # Check if parameter count decreased significantly from initial count
        if self.initial_parameter_count > 0:
            decrease_ratio = (self.initial_parameter_count - current_param_count) / self.initial_parameter_count
            
            if decrease_ratio >= self.decrease_threshold:
                should_stop = True
                msg = f"Stopping: {self.metric_name} decreased by {decrease_ratio:.2%} from initial "
                msg += f"(from {self.initial_parameter_count} to {current_param_count})"
                if epoch is not None:
                    msg += f" at epoch {epoch}"
                print(msg)
        
        return should_stop
    
    def reset(self):
        """Reset the stopper state."""
        self.initial_parameter_count = None

class AccuracyAndReductionStopper(BaseStopper):
    """
    Stops training when both accuracy reaches target AND parameter count decreases by threshold.
    Uses AccuracyStopper and ParameterCountStopper internally.
    """
    def __init__(self, target_accuracy=0.9, parameter_decrease_threshold=0.5):
        super().__init__()
        self.accuracy_stopper = AccuracyStopper(target_accuracy=target_accuracy)
        self.parameter_stopper = ParameterCountStopper(decrease_threshold=parameter_decrease_threshold)
        self.accuracy_reached = False
        self.parameter_reduced = False
    
    def check(self, model, x_train, y_train, epoch=None, params=None):
        """
        Check if both accuracy target is reached AND parameter count decreased by threshold.
        
        Args:
            model: The neural network model
            x_train: Training input data
            y_train: Training target data
            epoch: Current epoch number (optional)
            
        Returns:
            bool: True if both conditions are met, False otherwise
        """
        if model is None or x_train is None or y_train is None:
            return False
        should_stop = False
        # Check accuracy condition
        self.accuracy_reached = self.accuracy_stopper.check(model, x_train, y_train, epoch, params)
        
        # Check parameter reduction condition
        self.parameter_reduced = self.parameter_stopper.check(model, x_train, y_train, epoch, params)
        
        # Stop if both conditions are met
        if self.accuracy_reached and self.parameter_reduced:
            should_stop = True
            msg = f"Stopping: Both conditions met - accuracy >= {self.accuracy_stopper.target_accuracy:.2f} , {self.accuracy_reached:.2f} "
            msg += f"AND parameters reduced by >= {self.parameter_stopper.decrease_threshold:.1%}, {self.parameter_reduced:.2f}"
            if epoch is not None:
                msg += f" at epoch {epoch}"
            print(msg)
        
        return should_stop
    
    def reset(self):
        """Reset the stopper state."""
        super().reset()
        self.accuracy_stopper.reset()
        self.parameter_stopper.reset()
        self.accuracy_reached = False
        self.parameter_reduced = False
