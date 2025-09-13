import numpy as np
import os, sys


if len(sys.argv) != 3:
    print("Usage: python EigRes.py <N> <result root path>")
    exit(1)

N = int(sys.argv[1])
resDir = sys.argv[2]
subdirs = [x[0] for x in os.walk(resDir)][1:]

for d in subdirs:
    print(f"Process folder {d}")
    Vfiles = [f for f in os.listdir(d) if "V_" in f]
    Vfiles.sort()
    for i in range(len(Vfiles)):
        V = np.load(os.path.join(d, Vfiles[i]))
        
        ortho = V.T @ V - np.eye(V.shape[0])
        residual = np.linalg.norm(ortho, axis=0)
        print(residual.sum())
        np.save(os.path.join(d, f"Orth_{Vfiles[i].split('_')[1]}"), residual)
        del V

print("Finished.")