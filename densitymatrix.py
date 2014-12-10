import numpy as np
import pauli
import auxiliary as aux
import auxiliary_old as aux_old
from sarmaH import *
import multiprocessing
from time import clock

""" setting parameters """ # Made these random to track sparsity
ph=5 #number of photons
Alist=np.array([1.0,1.0,1.0,1.0]) #hyperfine coefficients
wlist=2*np.array([1.0,1.0,1.0,1.0]) #nuclear Zeeman
bmat=0.25*np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]) #dipolar coupling
Omega=3.0 #emitter Zeeman

#initial unpolarized state of environment, the ugly way
yp, ym=pauli.yp, pauli.ym
mkron = lambda x: reduce(np.kron, x)
envinit=1.0/6.0*(np.outer(np.conj(mkron((yp,yp,ym,ym)).T), mkron((yp,yp,ym,ym)))+ \
                 np.outer(np.conj(mkron((yp,ym,ym,yp)).T), mkron((yp,ym,ym,yp)))+ \
                 np.outer(np.conj(mkron((ym,ym,yp,yp)).T), mkron((ym,ym,yp,yp)))+ \
                 np.outer(np.conj(mkron((yp,ym,yp,ym)).T), mkron((yp,ym,yp,ym)))+ \
                 np.outer(np.conj(mkron((ym,yp,ym,yp)).T), mkron((ym,yp,ym,yp)))+ \
                 np.outer(np.conj(mkron((ym,yp,yp,ym)).T), mkron((ym,yp,yp,ym))))

def getdm_new(x): return aux.dmat(ph,Omega,wlist,Alist,bmat,envinit)

def getdm_old(x): return aux_old.dmat(ph,Omega,wlist,Alist,bmat,envinit)

t=clock()
dms=getdm_new(0)
t2=clock()-t
print(t2)

dmsred=aux.TraceOverEnv(dms,4)
Z1=pauli.spspec(ph+1,3,1)
X2=pauli.spspec(ph+1,1,2)
Z3=pauli.spspec(ph+1,3,3)
observable=reduce(np.dot,[Z1,X2,Z3])
ZXZ=np.trace(np.dot(observable,dmsred))

projectors=[pauli.zp,pauli.xp,pauli.zp]
tobmsrd=[1,2,3]
for i in range(3):
    dmsred=aux.measurement(dmsred,tobmsrd[i],projectors[i])

#trace over dot
dmatphotons=aux.TraceOverDot(dmsred)
eof=aux.eof(dmatphotons)
print(eof)
