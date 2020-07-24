try:
    import cupy as cp
except ImportError:
    warning_msg = ("\033[01mWARNING\033[m: Module \033[03mcupy\033[m is not "
                   "installed on this system. \033[03mcupy\033[m enables GPU "
                   "based acceleration through multiprocessing.")
    print(warning_msg)

from .core import *
from .lib import *
from .utils import *
