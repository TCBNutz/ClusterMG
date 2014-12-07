import numpy as np
import itertools as it
from time import clock

Fb=[1]*2**ph
identity=np.diag([1]*envdim)


for j in range(2**ph):
    for i in range(ph):
        if i==0:
            Fb[j]=np.dot(A[b[j][ph-1],0],identity)
        else:
            Fb[j]=np.dot(A[b[j][ph-i-1],b[j][ph-i]],Fb[j])
