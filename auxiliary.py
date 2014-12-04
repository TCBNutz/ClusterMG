""" some auxiliary functions """
import itertools
import numpy as np

""" function to give state of the unmeasured qubits after projective measurement
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


