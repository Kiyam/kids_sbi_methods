import numpy as np
import scipy as sc
import configparser as cfg
import subprocess
import sys
import os
import glob
import tarfile
import shutil
import errno
import re
import kcap_methods
import kcap_methods.score_compression
from icecream import ic
from environs import Env

class kcap_delfi(kcap_methods.kcap_methods.kcap_deriv):
    def __init__(self):
        pass