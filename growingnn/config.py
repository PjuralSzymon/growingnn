import os
error_clip_range = 600 # In back propagation we are clipping the error range to prevent exploding gradients
WEIGHTS_CLIP_RANGE = 400 # After weight update we are clipping all weights to prevent exploding gradients
LARGE_MAX = 2**128
MAX_THREADS = os.cpu_count() * 5