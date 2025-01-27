import os
from enum import Enum

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

