""" 
Pauli Matrices and Hamiltonian construction 
Note from Pete: I've worked through this and optimized it a bit.
Still some #TODOs left
"""
import numpy as np
#import errorcounting as erc

# These define pauli operators
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1J],[1J,0]])
sz = np.array([[1,0],[0,-1]])
s0 = np.array([[1,0],[0,1]])
sp = 0.5*(sx+1J*sz)
sm = 0.5*(sx-1J*sz)
svec = [s0,sx,sy,sz,sp,sm]
s01 = np.array([[0,1],[0,0]])
s10 = np.array([[0,0],[1,0]])
xp = (1.0/np.sqrt(2))*np.array([1,1])
xm = (1.0/np.sqrt(2))*np.array([1,-1])
yp = (1.0/np.sqrt(2))*np.array([1,1J])
ym = (1.0/np.sqrt(2))*np.array([1,-1J])
zp = np.array([1,0])
zm = np.array([0,1])

# Here we define some useful matrix operations
def contran(a):
    """ Gives the conjugate transpose of the matrix a """
    return np.conjugate(np.transpose(a))

def spn(n, typ):
    """ Gives the operator type:(s0,sx,sy,sz,sp,sm) to the kronecker product power n"""
    return reduce(np.kron, (svec[typ] for i in xrange(n)))

def spspec(n, typ, wh):
    """ Of n total spins, this gives type typ acting on spin wh """
    return reduce(np.kron, (svec[typ] if i==wh else svec[0] for i in xrange(n)))

def allp(n, typ):
    """ Of n total spins, this gives the sum of type typ on each, 
    i.e \sum_i \sigma_typ^(i) """
    mat = np.zeros((pow(2,n),pow(2,n)))
    for i in range(n):
        mat = mat+spspec(n,typ,i)
    return mat
    
def allJ2(n):
    """ Gives total J^2 operator (with factors of 2),
    i.e. (\sum_i \sigma_x^(i))^2 +(\sum_i \sigma_y^(i))^2 +(\sum_i \sigma_y^(i))^2 """
    x = allp(n,1)/2.
    y = allp(n,2)/2.
    z = allp(n,3)/2.
    return np.dot(x,x)+np.dot(y,y)+np.dot(z,z)

def allpwieght(n,typ,ws):
    """ Of n spins, this gives the sum of type typ acting on each, wieghted by ws,
    i.e \sum_i ws_i \sigma_typ^(i)""" #TODO
    mat = np.zeros((pow(2,n),pow(2,n)))
    for i in range(n):
        mat = mat+spspec(n,typ,i)*ws[i]
    return mat
    
def CompFromBinary(b1,b2,wbasis):
    """ Generates the matrix |b1><b2| from the binary strings b1 and b2 in the basis wbasis 
    e.g. gives [[1,0],[0,0]] from strings [1,0] and [1,0] in z basis""" #TODO
    bs = [[],[xp,xm],[yp,ym],[zp,zm]][wbasis]
    blen = len(b1)
    vec1 = [1]
    vec2 = [1]
    for i in range(blen):
        vec1 = np.kron(vec1,bs[int(b1[i])])
    for j in range(blen):
        vec2 = np.kron(vec2,bs[int(b2[j])])
    return np.outer(vec1,contran(vec2))
    
def CompFromBinaryMatrix(binmat,n,m,wbasis):
    """ Generates the matrix in the computational basis from a matrix binmat 
    which is in the binary basis with fixed m=total projection, n in the number of spins""" #TODO
    dim = len(binmat)
    mat = np.zeros([2**n,2**n])
    basises = erc.basisnm(n,m)
    for i in range(dim):
        for j in range(dim):
            mat = mat + binmat[i,j]*CompFromBinary(basises[i],basises[j],wbasis)
    return mat
    
if __name__=='__main__':
    print allJ2(1)
