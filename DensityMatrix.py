import itertools
import numpy as np
import pauli
import auxiliary as aux

""" setting parameters """
ph=2#number of photons
Alist=np.array([0,0,0,0])#hyperfine coefficients
wlist=np.array([0,0,0,0]) #nuclear Zeeman
bmat=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]) #dipolar coupling
Omega=15 #emitter Zeeman

#initial unpolarized state of environment, the ugly way
envinit=1.0/6.0*(np.outer(np.conj(np.kron(pauli.yp,np.kron(pauli.yp,np.kron(pauli.ym, pauli.ym))).T),np.kron(pauli.yp,np.kron(pauli.yp,np.kron(pauli.ym, pauli.ym))))+ \
                     np.outer(np.conj(np.kron(pauli.yp,np.kron(pauli.ym,np.kron(pauli.ym, pauli.yp))).T),np.kron(pauli.yp,np.kron(pauli.ym,np.kron(pauli.ym, pauli.yp))))+ \
                     np.outer(np.conj(np.kron(pauli.ym,np.kron(pauli.ym,np.kron(pauli.yp, pauli.yp))).T),np.kron(pauli.ym,np.kron(pauli.ym,np.kron(pauli.yp, pauli.yp))))+ \
                     np.outer(np.conj(np.kron(pauli.yp,np.kron(pauli.ym,np.kron(pauli.yp, pauli.ym))).T),np.kron(pauli.yp,np.kron(pauli.ym,np.kron(pauli.yp, pauli.ym))))+ \
                     np.outer(np.conj(np.kron(pauli.ym,np.kron(pauli.yp,np.kron(pauli.ym, pauli.yp))).T),np.kron(pauli.ym,np.kron(pauli.yp,np.kron(pauli.ym, pauli.yp))))+ \
                     np.outer(np.conj(np.kron(pauli.ym,np.kron(pauli.yp,np.kron(pauli.yp, pauli.ym))).T),np.kron(pauli.ym,np.kron(pauli.yp,np.kron(pauli.yp, pauli.ym)))))


""" tracing over the environment to obtain the reduced density matrix dmatred"""
dmatred=np.array([[0.+0.j]*2**(ph+1)]*2**(ph+1))
envdim=2**len(Alist) #environment dimension
for m in range(2**(ph+1)):
    for n in range(2**(ph+1)):
        dmatred[m,n]=np.trace(aux.dmat(ph,Omega,wlist,Alist,bmat,envinit)[m*envdim:(m+1)*envdim,n*envdim:(n+1)*envdim])

print(dmatred)
