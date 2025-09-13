import numpy as np
import scipy.io
import os, sys
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: python BatchNpy2Mat.py <npy file name> <mat file name>")
    exit(1)

npyFn = sys.argv[1]
matFn = sys.argv[2]
M = np.load(npyFn)
scipy.io.savemat(matFn, dict(data=M))
print("Done.")