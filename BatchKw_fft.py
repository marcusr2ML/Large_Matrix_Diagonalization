import numpy as np
import os, sys


def spec_kw_fft_ph(omega, V, E ,ee):
    N = int(np.sqrt(V.shape[0] / 2))
    u = V[:N**2, :]
    v = V[N**2:, :]
    u = u.reshape((N, N, V.shape[1]))
    v = v.reshape((N, N, V.shape[1]))
    d = 8;
    N = N-2*d;
    u = u[d:-d,d:-d,:]
    v = v[d:-d,d:-d,:]	
    L1 = ee / ((omega - E) ** 2 + ee ** 2) / np.pi
    L2 = ee / ((omega + E) ** 2 + ee ** 2) / np.pi

    u_fft = np.fft.fft2(u, axes=(0, 1))
    v_fft = np.fft.fft2(v, axes=(0, 1))

    u_fft = np.abs(u_fft) ** 2
    v_fft = np.abs(v_fft) ** 2

    u_fft = u_fft.reshape((N**2, V.shape[1]))
    v_fft = v_fft.reshape((N**2, V.shape[1]))

    h = np.fft.fftshift((u_fft @ L1).reshape((N, N))) / (N**2)
    p = np.fft.fftshift((v_fft @ L2).reshape((N, N))) / (N**2)
    
    return p, h

def spec_kw_fft_w(V, E, ee):
    N = int(np.sqrt(V.shape[0] / 2))
    w = np.linspace(-1.5, 1.5, 501)

    u = V[:N**2, :]
    v = V[N**2:, :]
    u = u.reshape((N, N, V.shape[1]))
    v = v.reshape((N, N, V.shape[1]))

    d = 8;
    N = N-2*d;
    u = u[d:-d,d:-d,:]
    v = v[d:-d,d:-d,:]

    u_fft = np.fft.fft2(u, axes=(0, 1))
    v_fft = np.fft.fft2(v, axes=(0, 1))

    u_fft = np.abs(u_fft) ** 2
    v_fft = np.abs(v_fft) ** 2

    u_fft = u_fft.reshape((N**2, V.shape[1]))
    v_fft = v_fft.reshape((N**2, V.shape[1]))

    xv, yv = np.meshgrid(w, E)

    wp = xv + yv
    wm = xv - yv

    L2 = ee / (wp**2 + ee**2) / np.pi
    L1 = ee / (wm**2 + ee**2) / np.pi

    h = np.fft.fftshift((u_fft @ L1).reshape((N, N, w.shape[0]))) / (N**2)
    p = np.fft.fftshift((v_fft @ L2).reshape((N, N, w.shape[0]))) / (N**2)

    return p, h

if len(sys.argv) != 5:
    print("Usage: python BatchKw_fft.py <N> <result root path> <output name> <ee>")
    exit(1)

N = int(sys.argv[1])
d = 8
N = N -2*d
omega = 0
resDir = sys.argv[2]
outputFn = sys.argv[3]
ee = float(sys.argv[4])

subdirs = [x[0] for x in os.walk(resDir)][1:]

pAcc = np.zeros((N, N), dtype=np.float64)
hAcc = np.zeros((N, N), dtype=np.float64)
pwAcc = np.zeros((N, N, 501), dtype=np.float64)
hwAcc = np.zeros((N, N, 501), dtype=np.float64)
for d in subdirs:
    print(f"Process folder {d}")
    Efiles = [f for f in os.listdir(d) if "E_" in f]
    Vfiles = [f for f in os.listdir(d) if "V_" in f]
    Efiles.sort()
    Vfiles.sort()
    for i in range(len(Efiles)):
        E = np.load(os.path.join(d, Efiles[i]))
        V = np.load(os.path.join(d, Vfiles[i]))
        p, h = spec_kw_fft_ph(omega, V, E, ee)
        pw, hw = spec_kw_fft_w(V, E, ee)
        pAcc += p
        hAcc += h
        pwAcc += pw
        hwAcc += hw

        del E
        del V

np.save("p_" + outputFn, pAcc)
np.save("h_" + outputFn, hAcc)
np.save("pw_" + outputFn, pwAcc)
np.save("hw_" + outputFn, hwAcc)

print("Finished.")
