from .action import *
from .painter import *
from .structure import *
from .trainer import *
from .helpers import *
from .config import *
from .quaziIdentity import *

IS_CUPY = False

def switch_to_gpu():
   # print(" parent: switch_to_gpu")
    global np, IS_CUPY
    try:
        helpers.switch_to_gpu()
        structure.switch_to_gpu()
        import cupy as np
        IS_CUPY = False
        print(f"Cupy library loaded, GPU enabled")
    except Exception as e:
        print(f"Unexpected error occurred: {e} while loading cupy library, switching to CPU")
        switch_to_cpu()  # Fallback to CPU

def switch_to_cpu():
#    print(" parent: switch_to_cpu")
    global IS_CUPY, np
    helpers.switch_to_cpu()
    structure.switch_to_cpu()
    import numpy as np
    IS_CUPY = True
    print("Numpy library loaded, CPU enabled.")

switch_to_cpu()