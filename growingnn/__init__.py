# Configure NumPy/OpenBLAS to use all CPU cores (must be BEFORE numpy import)
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count()))  # Use all cores
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))       # OpenMP fallback

from .action import *
from .painter import *
from .structure import *
from .trainer import *
from .helpers import *
from .config import config, DistributionMode
from .quaziIdentity import *
from .utils import Loss, Activations, LearningRateScheduler, History, Storage

# GPU/CuPy functionality removed - using CPU only
import numpy as np