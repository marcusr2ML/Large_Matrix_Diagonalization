import numpy as np
import os, sys
import scipy
from scipy.sparse import kron

def GetHermitian(N, S, delta_o, precision=np.float64):

    def kron3(A,B,C):
        D= kron(A,kron(B,C))
        return D

    sz=np.array([[1.0,0],[0,-1]])
    sx=np.array([[0,1.0],[1,0]])
    sy=np.array([[0,1.0j],[-1j,0]])

    if precision == np.float64:
        cprecision = np.cdouble
    else:
        cprecision = np.csingle

    sz = scipy.sparse.csr_matrix(sz, dtype=precision)
    sx = scipy.sparse.csr_matrix(sx, dtype=precision)
    sy = scipy.sparse.csr_matrix(sy, dtype=cprecision)

    # N=50

    t0 = np.eye(N)
    t1 = np.diag(np.ones(N - 1),k=1) + np.diag(np.ones(N - 1),k=-1)
    t1[0,-1]=1
    t1[-1,0]=1
    tt1 = np.diag(np.ones(N - 2),k=2) + np.diag(np.ones(N - 2),k=-2)
    tt1[0,-2]=1
    tt1[-2,0]=1
    tt1[1,-1]=1
    tt1[-1,1]=1

    ###############
    t =0
    tt = 1
    ttt = 0
    tttt=0
    mu = 0

    t0 = np.array(t0, dtype=precision)
    t1 = np.array(t1, dtype=precision)
    tt1 = np.array(tt1, dtype=precision)

    t0 = scipy.sparse.csr_matrix(t0)
    t1 = scipy.sparse.csr_matrix(t1)
    tt1 = scipy.sparse.csr_matrix(tt1)


    t = 0.25
    tt = -0.031863
    ttt = 0.016487
    tttt = 0.0076112
    mu = -0.16235
    ###############
    # T = -mu*kron(t0,t0)\
    #     -t*(kron(t1,t0)+kron(t0,t1))\
    #     -tt*kron(t1,t1)-ttt*(kron(tt1,t0)+kron(t0,tt1))\
    #     -tttt*kron(tt1,tt1)
    T = -mu*kron3(sz,t0,t0)\
    -t*(kron3(sz,t1,t0)+kron3(sz,t0,t1))\
    -tt*kron3(sz,t1,t1)-ttt*(kron3(sz,tt1,t0)+kron3(sz,t0,tt1))\
    -tttt*kron3(sz,tt1,tt1)

    ###########################
    # delta_o=.06     #This can be played around

    dform = 1/2*kron(t1,t0)- kron(t0,t1)

    S = delta_o * scipy.sparse.diags(S.flatten(), dtype=cprecision)
    S = S @ dform + dform @ S
    S = scipy.sparse.vstack(
        [scipy.sparse.hstack([scipy.sparse.csr_matrix((N**2, N**2)), S]),
        scipy.sparse.hstack([S.getH(), scipy.sparse.csr_matrix((N**2, N**2))])],
        dtype=cprecision
        )
    T += S
    print(T.dtype)
    print(f'Hermitian Generated. Shape: ' + str(T.shape))
    return scipy.sparse.csr_matrix(T)

if len(sys.argv) != 3:
    print("Usage: python EigRes.py <N> <result root path> <output name>")
    exit(1)

N = int(sys.argv[1])
resDir = sys.argv[2]
subdirs = [x[0] for x in os.walk(resDir)][1:]

delta_o = 0.06
Sr = np.loadtxt('re_S_frac_N=400_r0=16')
Si = np.loadtxt('im_S_frac_N=400_r0=16')
S = Sr + Si * 1j

T = GetHermitian(N, S, delta_o)

for d in subdirs:
    print(f"Process folder {d}")
    Efiles = [f for f in os.listdir(d) if "E_" in f]
    Vfiles = [f for f in os.listdir(d) if "V_" in f]
    Efiles.sort()
    Vfiles.sort()
    for i in range(len(Efiles)):
        E = np.load(os.path.join(d, Efiles[i]))
        V = np.load(os.path.join(d, Vfiles[i]))
        
        residual = T @ V - V * E
        residual = np.linalg.norm(residual, axis=0)
        print(residual.sum())
        np.save(os.path.join(d, f"Res_{Efiles[i].split('_')[1]}"), residual)
        del E
        del V

print("Finished.")
