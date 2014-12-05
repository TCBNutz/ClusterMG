'''
Contructing the Das Sarma pure-dephasing Hamiltonian in full Hilbert Space
L. Cywinski, W. M. Witzel, and S. Das Sarma, PRL 102, 057601 (2009)
L. Cywinski, W. M. Witzel, and S. Das Sarma, PRB 79, 245314 (2009)
'''

from pauli import *
from scipy import linalg
import numpy as np

def SDip(n,  bmat):
    """ Generates the nuclear dipolar coupling in the system+environment Hilbert space:
    n is number of spins in the bath
    b is the coupling between bath spins,  which should have zero diagonal"""
    mat = np.zeros((pow(2, n), pow(2, n)))
    for i in range(n):
        for j in range(n):
            mat = mat + 0.25*bmat[i, j]*(np.dot(spspec(n, 4, i), spspec(n, 5, j))-2*np.dot(spspec(n, 2, i), spspec(n, 2, j)))
    return np.kron(s0, mat)
    
def SY(n, wlist, Omega):
    """ Gives the Zeeman Hamiltonian in the y-basis"""
    sys = Omega*np.kron(sy, np.identity(pow(2, n)))/2.
    bath = np.kron(s0, allpwieght(n, 2, wlist))/2.
    return sys + bath
    
def SHI(n, Alist):
    """ Gives the pure-dephasing interaction of the central spin and the environment """
    return 0.25*np.kron(sy, allpwieght(n, 2, Alist))

def makeAmat(Alist):
    """ This generates the matrix with elements A_i A_j from the vector with elements A_i """
    n = len(Alist)
    Amat = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Amat[i, j] = Alist[i]*Alist[j]
    Amat = Amat - np.dot(np.diag(Alist), np.diag(Alist))
    return Amat
    
def SH2(n, Alist, Omega):
    """ This is the hf-mediated two-spin flip-flop interaction, given in eq. 20 of Cywinski B 79 paper """
    Amat = makeAmat(Alist)
    bath = np.zeros((pow(2, n), pow(2, n)))
    for i in range(n):
        for j in range(n):
            bath = bath + 0.125*(Amat[i, j]/(2*Omega))*np.dot(spspec(n, 4, i), spspec(n, 5, j))
    return np.kron(sy, bath)
    
def Hhf(n, Alist):
    """ Gives the full spin-environment hyper-fine interaction term """
    return 0.25*np.kron(sy, allpwieght(n, 2, Alist))+0.25*np.kron(sx, allpwieght(n, 1, Alist))+0.25*np.kron(sz, allpwieght(n, 3, Alist))
    
def STot(n, Alist, wlist, bmat, Omega):
    """ Adds together all terms in the Hamiltonian
    Alist is a vector of coupling strengths to each bath spin
    wlist is a list of zeeman energies for the environment
    bmat is a matrix of dipolar coupling constants between the bath spins"""
    return SY(n, wlist, Omega)+SHI(n, Alist)+SH2(n, Alist, Omega)+SDip(n, bmat)
    
def STothf(n, Alist, wlist, bmat, Omega):
    """ Adds together all terms in the full hyper-fine Hamiltonian
    Alist is a vector of coupling strengths to each bath spin
    wlist is a list of zeeman energies for the environment
    bmat is a matrix of dipolar coupling constants between the bath spins"""
    return SY(n, wlist, Omega)+Hhf(n, Alist)+SDip(n, bmat)
    
# What follows constructs the Hamiltonian for specific forms of the Alist,  wlist and bmat
    
def ADist(AT, n):
    """ Gives a list of values decaying in a Guassian fashion.
    AT is the sum of all the n values """
    Alist = np.array([])
    for i in range(n):
        Alist = np.append(Alist, np.exp(-((i)/(0.5*n))**2))
    return AT*Alist/np.sum(Alist)
    
def Bmat(BT, n):
    """ Makes a Bmat with elements b_i b_j where b is a vector generated from ADist above """
    blist = ADist(BT, n)
    return makeAmat(blist)
    
def BmatCon(BT, n):
    """ Makes a constant matrix with zero diagonal and sum of all elements BT """
    blist = np.ones(n)
    bmat =BT*makeAmat(blist)/(n*(n-1))
    return bmat
    
def STotSpec(n, AT, BT, WT, Omega):
    """ Gives the complete Sarma Hamiltonian in terms of totals AT,  AT,  WT
    Alist is distributed
    wlist is constant with total WT
    bmat is distributed similarly to Alist """
    alist = ADist(AT, n)
    wlist = WT*np.ones(n)/n
    bmat = Bmat(BT, n)
    return STot(n, alist, wlist, bmat, Omega)
    
def SEvol(n, Alist, wlist, bmat, Omega, t):
    """ Exponentiates the full Hamiltonian given full coupling constants """
    return linalg.expm2(-1J*t*STot(n, Alist, wlist, bmat, Omega))
    
def SEvolSpec(n, AT, BT, WT):
    """ Exponentiates Hamiltonian given total coupling constants """
    alist = ADist(AT, n)
    wlist = WT*np.ones(n)/n
    bmat = Bmat(BT, n)
    return SEvol(n, alist, wlist, bmat, 1, np.pi/2)
    
def SEvolSpechf(n, AT, BT, WT):
    """ Exponentiates hyper-fine Hamiltonian given total coupling constants """
    alist = ADist(AT, n)
    wlist = WT*np.ones(n)/n
    bmat = Bmat(BT, n)
    return linalg.expm2((-1J*np.pi/2)*STothf(n, alist, wlist, bmat, 1))
    
def SEvolUniformhf(n, AT, BT, WT):
    """ Exponentiates hyper-fine Hamiltonian given total coupling constants,  
    assuming uniform coupling to environment spins """
    alist = AT*np.ones(n)/n
    wlist = WT*np.ones(n)/n
    bmat = Bmat(BT, n)
    return linalg.expm2((-1J*np.pi/2)*STothf(n, alist, wlist, bmat, 1))
    
def UnPolVecYSym(n):
    """ Gives the least polarised state of n spins in the y direction,  
    but with high symmetric,  i.e. |000...0111, , , 1>"""
    vec = 1
    for i in range(n/2):
        vec = np.kron(vec, yp)
    for i in range(n/2):
        vec = np.kron(vec, ym)
    return vec
    
def UnPolVecY(n):
    """ Gives the least polarised state of n spins in the y direction, 
    i.e. |0101...01> """
    evecs = [yp, ym]
    vec = 1
    for i in range(n):
        vec = np.kron(vec, evecs[np.mod(i, 2)])
    return vec
    
def UnPolMatY(n):
    """ Gives the least polarised pure density operator of n spins in the y direction """
    vec = UnPolVecY(n)
    return np.outer(vec, contran(vec))
    
def UnPolMatYSym(n):
    """ Gives the least polarised pure density operator of n spins in the y direction,  
    but with high symmetry as described above """
    vec = UnPolVecYSym(n)
    return np.outer(vec, contran(vec))

def SortedESys(n):
    """ Returns eigen system of total S_y operator sorted such that states with least 
    absolute eigenvalue come first """
    matrix = allp(n, 2)
    evals, evecs = linalg.eig(matrix)
    evecsnew = np.transpose(evecs)
    ind = abs(evals).argsort()
    evalsnew = evals[ind]
    evecsnewnew = evecsnew[ind]
    return [evalsnew, evecsnewnew]

def LeastPolMat(n):
    """ Of n spins gives a mixed state of the two states with lowest total S_y """
    evecs = SortedESys(n)[1]
    evec0 = evecs[0]
    evec1 = evecs[1]
    mat0 = np.outer(evec0, contran(evec0))
    mat1 = np.outer(evec1, contran(evec1))
    return 0.5*mat0+0.5*mat1
    
def ZeroEvecs(n):
    """ Gives the eigenvectors of total S_y operator with zero eigenvalue """
    matrix = allJ(n)
    evals, evecs = linalg.eig(matrix)
    evecst = np.transpose(evecs)
    sevecs = [np.zeros(2**n)]
    for i in range(len(evals)):
        if np.round(evals[i], 5)==0:
            sevecs = np.append(sevecs, [evecst[i]], 0)
    return sevecs

def ZeroJMat(n):
    """ Gives a even mixed state of states with zero total S_y """
    vecsmat = ZeroEvecs(n)
    many = len(vecsmat)-1
    dmat = np.zeros([2**n, 2**n])
    for i in range(many):
        dmat = dmat + np.outer(vecsmat[i+1], contran(vecsmat[i+1]))/many
    return dmat
    
