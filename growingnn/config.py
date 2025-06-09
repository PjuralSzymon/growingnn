import os
from enum import Enum
import numpy

class DistributionMode(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'
    GAMMA = 'gamma'
    REVERSED_GAUSSIAN = 'reversed_gaussian'

class Config:
    # Weight and Distribution Settings
    WEIGHT_DISTRIBUTION_MODE = DistributionMode.NORMAL
    WEIGHTS_CLIP_RANGE = 3
    #weights_clip_range = 400
    LARGE_MAX = 2**128
    
    # Neural Network Settings
    FLOAT_TYPE = numpy.float64
    VERSION = 'R3.3'
    MAX_THREADS = max(1, int(os.cpu_count() * 0.5))
    
    # Training Settings
    ERROR_CLIP_RANGE = 600
    PROGRESS_PRINT_FREQUENCY = 7
    
    # Feature Flags
    THROW_EXCEPTION = True
    SAVE_PLOTS = True
    ENABLE_CLIP_ON_OPTIMIZERS = False
    ENABLE_CLIP_ON_ACTIVATIONS = False
    
    # Neural Network Structure Settings
    MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL = 3
    MINIMUM_MATRIX_SIZE_FOR_CONNECTIONS_REMOVAL = 3
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration values at runtime"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Configuration '{key}' does not exist")
    
    @classmethod
    def get(cls, key, default=None):
        """Get configuration value with optional default"""
        return getattr(cls, key, default)
    
    @classmethod
    def reset_to_defaults(cls):
        """Reset all configuration values to their defaults"""
        cls.WEIGHT_DISTRIBUTION_MODE = DistributionMode.NORMAL
        cls.WEIGHTS_CLIP_RANGE = 3
        cls.weights_clip_range = 400
        cls.LARGE_MAX = 2**128
        cls.FLOAT_TYPE = numpy.float64
        cls.VERSION = 'R3'
        cls.MAX_THREADS = max(1, int(os.cpu_count() * 0.5))
        cls.ERROR_CLIP_RANGE = 600
        cls.PROGRESS_PRINT_FREQUENCY = 7
        cls.THROW_EXCEPTION = True
        cls.SAVE_PLOTS = True
        cls.ENABLE_CLIP_ON_OPTIMIZERS = False
        cls.ENABLE_CLIP_ON_ACTIVATIONS = False
        cls.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL = 3
        cls.MINIMUM_MATRIX_SIZE_FOR_CONNECTIONS_REMOVAL = 3

# Create a global instance for backward compatibility
config = Config()

# Example usage:
# from growingnn.config import config
# 
# # Update multiple settings
# update(
#     WEIGHTS_CLIP_RANGE=5,
#     PROGRESS_PRINT_FREQUENCY=10,
#     ENABLE_CLIP_ON_ACTIVATIONS=True
# )
# 
# # Get a setting with default
# batch_size = get('BATCH_SIZE', 32)
# 
# # Reset to defaults
# reset_to_defaults()
