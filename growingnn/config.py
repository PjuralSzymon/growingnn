import os
from enum import Enum
import numpy

class DistributionMode(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'
    GAMMA = 'gamma'
    REVERSED_GAUSSIAN = 'reversed_gaussian'
    
WEIGHT_DISTRIBUTION_MODE = DistributionMode.NORMAL
error_clip_range = 600 # In back propagation we are clipping the error range to prevent exploding gradients
WEIGHTS_CLIP_RANGE = 3 # After weight update we are clipping all weights to prevent exploding gradients
weights_clip_range = 400 # After weight update we are clipping all weights to prevent exploding gradients
LARGE_MAX = 2**128
MAX_THREADS = 1#max(1, int(os.cpu_count() * 0.5))
VERSION = 'R3'
FLOAT_TYPE = numpy.float64

# Exception handling configuration
THROW_EXCEPTION = True  # Set to False to handle errors silently

# Progress printing configuration
PROGRESS_PRINT_FREQUENCY = 7  # Print progress every N epochs

# Add this with other configuration constants
SAVE_PLOTS = True  # Set to False in tests to disable plot saving

ENABLE_CLIP_ON_OPTIMIZERS = False # Switch to True reduce a risk of exploding gradients but it will increase a time of training by 21%

ENABLE_CLIP_ON_ACTIVATIONS = False # Switch to True reduce a risk of exploding gradients but it will increase a time of training by 21%
