import itertools
import numpy as np
from sarmaH import *
import pauli
import auxiliary as aux

""" setting parameters """
ph=2#number of photons
Alist=np.array([0,0,0,0])#hyperfine coefficients
envdim=2**len(Alist) #environment dimension
wlist=np.array([0,0,0,0]) #nuclear Zeeman
bmat=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]) #dipolar coupling
Omega=15 #emitter Zeeman

""" OmegaEff gives the effective magnetic field, we choose an average Overhauser field of zero """
OmegaEff=Omega + 0.25*sum(Alist**2)/Omega

""" constructing the pure dephasing Hamiltonian """
HamiltonianPD=STot(len(Alist),Alist,wlist,bmat,Omega)

""" eigensystem of HamiltonianPD in the form
(array([eig.val.1, eig.val.2,...]), array([eigenvector1,eigenvector2,...]))"""
eigsys=np.linalg.eigh(HamiltonianPD)

"""propagator of HamiltonianPD for t/hbar = Pi/(2 OmegaEff) in its eigenbasis,
i.e as diagonal matrix of exponentiated eigenvalues"""
Udiag=np.diag(np.exp(-1j*0.5*np.pi*eigsys[0]/OmegaEff))

""" propagator in the numberbasis """
Unum=np.dot(eigsys[1],np.dot(Udiag,np.conj(eigsys[1].T)))

""" Dara's A00,A01,A10,A11 environment operators """
U=np.array([[Unum[0:envdim,0:envdim],Unum[0:envdim,envdim:]],[Unum[envdim:,0:envdim],Unum[envdim:,envdim:]]])

b=list(itertools.product([0,1],repeat=ph)) #all possible bit strings with ph bits

Fb=[1]*2**ph #list of matrices giving the environment operators F(b) for each bit string b

identity=np.diag([1]*envdim)

""" creating Fb as a list of np.array objects, each being a matrix acting on the environment
corresponding to the operator F(\vec(b) in Dara's paper eq. 2) """
for j in range(2**ph):
    for i in range(ph):
        if i==0:
            Fb[j]=np.dot(U[b[j][ph-1],0],identity)
        else:
            Fb[j]=np.dot(U[b[j][ph-i-1],b[j][ph-i]],Fb[j])


""" density matrix of initial state of the environment, chosen as mix of all states with the minimum
absolute value of total S_y. Might use Dara's solution from sarmaH later, but just do it by brute
force for four environment spins here"""
envinit=1.0/6.0*(np.outer(np.conj(np.kron(pauli.yp,np.kron(pauli.yp,np.kron(pauli.ym, pauli.ym))).T),np.kron(pauli.yp,np.kron(pauli.yp,np.kron(pauli.ym, pauli.ym))))+ \
                 np.outer(np.conj(np.kron(pauli.yp,np.kron(pauli.ym,np.kron(pauli.ym, pauli.yp))).T),np.kron(pauli.yp,np.kron(pauli.ym,np.kron(pauli.ym, pauli.yp))))+ \
                 np.outer(np.conj(np.kron(pauli.ym,np.kron(pauli.ym,np.kron(pauli.yp, pauli.yp))).T),np.kron(pauli.ym,np.kron(pauli.ym,np.kron(pauli.yp, pauli.yp))))+ \
                 np.outer(np.conj(np.kron(pauli.yp,np.kron(pauli.ym,np.kron(pauli.yp, pauli.ym))).T),np.kron(pauli.yp,np.kron(pauli.ym,np.kron(pauli.yp, pauli.ym))))+ \
                 np.outer(np.conj(np.kron(pauli.ym,np.kron(pauli.yp,np.kron(pauli.ym, pauli.yp))).T),np.kron(pauli.ym,np.kron(pauli.yp,np.kron(pauli.ym, pauli.yp))))+ \
                 np.outer(np.conj(np.kron(pauli.ym,np.kron(pauli.yp,np.kron(pauli.yp, pauli.ym))).T),np.kron(pauli.ym,np.kron(pauli.yp,np.kron(pauli.yp, pauli.ym)))))

""" aux1 is a matrix such that every column is the state of the emitter corresponding to a bit
string. Hence the ith column is [[1],[0]] if the last bit in the ith bit string is 0 and
[[0],[1]] otherwise"""
aux1=np.array([[0]*2**ph,[0]*2**ph])
for i in range(2**ph):
    if b[i][0]==0:
        aux1[0,i]=1
    else:
        aux1[1,i]=1
            
""" aux2 is a matrix such that every column is the state of emitter and photonic bit string """
aux2=np.array([[[0]]*2**(ph+1)]*2**ph)
for i in range(2**ph):
    aux3=np.array([[0]]*2**ph)
    aux3[i,0]=1
    aux2[i]=np.kron(aux1[:,[i]],aux3)

"""full density matrix dmat"""
dmat=np.array([[0.]*2**(ph+1)*envdim]*2**(ph+1)*envdim)
for i in range(2**ph):
    for j in range(2**ph):
        dmat=dmat+np.kron(np.kron(aux2[i],aux2[j].T[0]),np.dot(Fb[i],np.dot(envinit,np.conj(Fb[j].T))))

""" dmat gives the (approximation to) the state |C_n> (eq. 1 in Dara's paper). To make this a Cluster
state we must rotate once more and then apply a Z-gate to each photon"""

"""Uph is the propagator for Pi/2 rotation in the emitter + photon string + environment number basis """
phidentity=np.diag([1.]*2**ph) #identity on photon string Hilbert space
Uph=np.kron(np.array([[1,0],[0,0]]),np.kron(phidentity,U[0,0]))+\
np.kron([[0,1],[0,0]],np.kron(phidentity,U[0,1]))+np.kron([[0,0],[1,0]],np.kron(phidentity,U[1,0]))+ \
np.kron([[0,0],[0,1]],np.kron(phidentity,U[1,1]))

"""Z-gate on each photon"""
PauliZ=np.array([[1,0],[0,-1]])
Zphi=[0,PauliZ]
for i in range(ph-1):
    Zphi[0]=np.kron(Zphi[1],PauliZ)
    Zphi[1]=Zphi[0]

Zph=Zphi[0] # this is Z x Z x ... Z (ph times)
ZphBig=np.kron(np.array([[1,0],[0,1]]),np.kron(Zph,identity))

dmat=np.dot(np.dot(ZphBig,Uph),np.dot(dmat,np.conj(np.dot(ZphBig,Uph).T)))

""" tracing over the environment to obtain the reduced density matrix dmatred """
dmatred=np.array([[0.+0.j]*2**(ph+1)]*2**(ph+1))
for m in range(2**(ph+1)):
    for n in range(2**(ph+1)):
        dmatred[m,n]=np.trace(dmat[m*envdim:(m+1)*envdim,n*envdim:(n+1)*envdim])

print(dmatred)
