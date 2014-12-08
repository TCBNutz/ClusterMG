import numpy as np
import itertools as it
from time import clock

ph=3
pdim=2**ph
b=np.array(list(it.product([0,1],repeat=ph)))
aux2=np.zeros((pdim, 2**(ph+1), 1))
aux1=np.vstack((b[:,0]==0, b[:,0]!=0))
for i in range(pdim):
    aux3=np.array([[0]]*pdim)
    aux3[i,0]=1
    aux2[i]=np.kron(aux1[:,[i]],aux3)
# aux1, aux2 and aux3 are very sparse binary matrices. there must be a good optimization
print aux2[:,:,0]
print aux1*1
