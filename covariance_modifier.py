import numpy as np
import os
import errno
import kcap_methods
import score_compression

# Some script to take a covariance and degrade it in some manner
# Ideas are to suppress the off diagonals, randomly change the magnitude of the on-diagonal components
