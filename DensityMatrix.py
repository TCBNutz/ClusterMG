import itertools
import numpy as np
from sarmaH import *

ph=4 #number of photons

""" Alist is a list of coefficients A_i giving the hyperfine interaction strengths with
environment spins J_i."""
Alist=np.array([0,0])#np.array([1,2])
envdim=2**len(Alist) #dimension of the environment of len(Alist) qubits

""" wlist is a list of nuclear Zeeman energies, denoted \omega_{\alpha [i]} in Cywinski paper"""
wlist=np.array([0,0])#[0.2,0.5])

""" bmat is a matrix of dipolar couplings of the nuclei, denoted b_{ij} in eq. 5 in Cywinski paper.
bmat is symmetric and has zero diagonal """
bmat=np.array([[0,0],[0,0]])#[[0,1],[1,0]])

""" Omega is the Zeeman energy of the emitter spin """
Omega=15

""" OmegaEff gives the effective magnetic field, we choose an average Overhauser field of zero """
OmegaEff=Omega + 0.25*sum(Alist**2)/Omega

""" constructing the pure dephasing Hamiltonian """
HamiltonianPD=STot(2,Alist,wlist,bmat,Omega)

""" eigensystem of HamiltonianPD in the form
(array([eig.val.1, eig.val.2,...]), array([eigenvector1,eigenvector2,...]))"""
eigsys=np.linalg.eigh(HamiltonianPD)

"""propagator of HamiltonianPD for t/hbar = Pi/(2 OmegaEff) in its eigenbasis,
i.e as diagonal matrix of exponentiated eigenvalues"""
Udiag=np.diag(np.exp(-1j*0.5*np.pi*eigsys[0]/OmegaEff))

""" propagator in the numberbasis """
Unum=np.dot(eigsys[1],np.dot(Udiag,np.conj(eigsys[1].T)))

A00=Unum[0:envdim,0:envdim]
A01=Unum[0:envdim,envdim:]
A10=Unum[envdim:,0:envdim]
A11=Unum[envdim:,envdim:]

""" propagator in block form """
U=[[A00,A01],[A10,A11]]

b=list(itertools.product([0,1],repeat=ph)) #all possible bit strings with ph bits

Fb=[1]*2**ph #list of matrices giving the environment operators F(b) for each bit string b

identity=np.diag([1]*envdim)

""" creating Fb as a list of np.array objects, each being a matrix acting on the environment
corresponding to the operator F(\vec(b) in Dara's paper eq. 2) """
for j in range(2**ph):
    Fb[j]=identity
    for i in range(ph):
        if i==0:
            Fb[j]=np.dot(U[b[j][i]][b[j][0]],Fb[j])
        else:
            Fb[j]=np.dot(U[b[j][i]][b[j][i-1]],Fb[j])


envinit=np.diag([1.]*envdim)/envdim #initial state of the environment, zero polarization

""" aux1 is a matrix such that every column is the state of the emitter corresponding to a bit
string. Hence the ith column is [[1],[0]] if the last bit in the ith bit string is 0 and
[[0],[1]] otherwise"""
aux1=np.array([[0]*2**ph,[0]*2**ph])
for i in range(2**ph):
    if b[i][ph-1]==0:
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
        
""" tracing over the environment to obtain the reduced density matrix dmatred """
dmatred=np.array([[0.+0.j]*2**(ph+1)]*2**(ph+1))
for m in range(2**(ph+1)):
    for n in range(2**(ph+1)):
        dmatred[m,n]=np.trace(dmat[m*envdim:(m+1)*envdim,n*envdim:(n+1)*envdim])

print(np.trace(dmatred))
