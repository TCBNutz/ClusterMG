""" some auxiliary functions """
import itertools
import numpy as np
from sarmaH import *
import pauli

""" function dmat gives full Emitter x PhotonString x Environment density matrix. ph is number of photons, Omega is
Zeeman energy of emitter, wlist is list of Zeeman energies of environment spins, Alist is list of hyperfine couplings"""
def dmat(ph,Omega,wlist,Alist,bmat,envinit):
    pdim=2**ph
    envdim=2**len(Alist) #environment dimension
    OmegaEff=Omega + 0.25*sum(Alist**2)/Omega #effective magnetic field, <Overhauser> = 0
    HamiltonianPD=STot(len(Alist),Alist,wlist,bmat,Omega) #pure dephasing Hamiltonian
    eigsys=np.linalg.eigh(HamiltonianPD) #(array([eig.val.1, eig.val.2,...]), array([eigenvector1,eigenvector2,...]))
    Udiag=np.diag(np.exp(-1j*0.5*np.pi*eigsys[0]/OmegaEff)) #propagator for t/hbar = Pi/(2 OmegaEff) in eigenbasis
    Unum=np.dot(eigsys[1],np.dot(Udiag,np.conj(eigsys[1].T))) #propagator in numberbasis

    #environment operators
    A=np.array([[Unum[:envdim,:envdim],Unum[:envdim,envdim:]],[Unum[envdim:,:envdim],Unum[envdim:,envdim:]]])

    #list of matrices Fb giving the environment operators F(b) for each bit string b
    b=np.array(list(itertools.product([0,1],repeat=ph)))
    Fb=np.ones((pdim, envdim, envdim), dtype=complex)
    identity=np.eye(envdim)
    #TODO: there must be a more efficient construction here 
    for j in range(pdim):
        for i in range(ph): # For each photon
            if i==0:
                Fb[j]=np.dot(A[b[j][ph-1],0],identity)
            else:
                Fb[j]=np.dot(A[b[j][ph-i-1], b[j][ph-i]], Fb[j]) # Act the operator for each photon

    # Making aux2 as a matrix such that every column is the state of emitter and photonic bit string
    aux2=np.zeros((pdim, 2**(ph+1), 1))
    for i in range(pdim):
        aux2[i, i if i<pdim/2 else pdim+i]=1

    # Precompute env dot Fb
    envfb = [np.dot(envinit, np.conj(Fb[j].T)) for j in xrange(pdim)]

    """full density matrix dmat"""
    d=2**(ph+1)*envdim
    dmatCn=np.zeros((d,d), dtype=complex)
    for i in range(pdim):
        for j in range(pdim):
            pre=np.kron(aux2[i], aux2[j].T[0])
            post=np.dot(Fb[i], envfb[j])
            region=np.kron(pre, np.ones(post.shape))
            x=np.arange(len(region))[np.sum(region, axis=0)>0]
            y=np.arange(len(region))[np.sum(region, axis=1)>0]
            # todo: need a generating function for this stuff
            print i, j
            print min(x), max(x)
            print min(y), max(y)
            print 
            dmatCn+=np.kron(pre, post)

    

    """ dmat gives the (approximation to) the state |C_n> (eq. 1 in Dara's paper). To make this a Cluster
    state we must rotate once more and then apply a Z-gate to each photon"""

    """Uph is the propagator for Pi/2 rotation in the emitter + photon string + environment number basis """
    phidentity=np.eye(pdim) #identity on photon string Hilbert space
    Uph=np.kron(np.array([[1,0],[0,0]]),np.kron(phidentity,A[0,0]))+\
    np.kron([[0,1],[0,0]],np.kron(phidentity,A[0,1]))+np.kron([[0,0],[1,0]],np.kron(phidentity,A[1,0]))+ \
    np.kron([[0,0],[0,1]],np.kron(phidentity,A[1,1]))

    """Z-gate on each photon"""
    Zph=reduce(np.kron, (pauli.sz for i in xrange(ph)))
    ZphBig=np.kron(np.array([[1,0],[0,1]]),np.kron(Zph,identity))
    return np.dot(np.dot(ZphBig,Uph),np.dot(dmatCn,np.conj(np.dot(ZphBig,Uph).T)))


""" function measurement returns state of the unmeasured qubits after projective measurement
onto |state> state of qubit at position position. Initial state given by dmat."""
def measurement(dmat,position,state):
    qb=int(np.round(np.log2(len(dmat[0])))) # number of qubits in initial state
    bitstring=list(itertools.product([0,1],repeat=qb-1))
    a=np.array([[0.+0.J]*2**(qb)]*2**(qb-1))
    for i in range(2**(qb-1)):
        b=[0]*qb
        for j in range(position):
            if bitstring[i][j]==0:
                b[j]=np.array([1,0])
            else:
                b[j]=np.array([0,1])
        b[position]=state
        for j in range(position,qb-1):
            if bitstring[i][j]==0:
                b[j+1]=np.array([1,0])
            else:
                b[j+1]=np.array([0,1])
        a[i]=reduce(np.kron,b)
    dmatnew=reduce(np.dot,[np.conj(a),dmat,a.T])
    return 1.0/np.trace(dmatnew)*dmatnew

