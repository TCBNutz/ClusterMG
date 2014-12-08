import numpy as np
import pauli
import auxiliary as aux
import auxiliary_old as aux_old
from sarmaH import *
import itertools
from time import clock

""" setting parameters """ # Made these random to track sparsity
ph=4 #number of photons
Alist=np.random.uniform(-1,1,4) #hyperfine coefficients
wlist=np.random.uniform(-1,1,4) #nuclear Zeeman
bmat=np.random.uniform(-1,1,(4,4)) #dipolar coupling
Omega=15 #emitter Zeeman

#initial unpolarized state of environment, the ugly way
yp, ym=pauli.yp, pauli.ym
mkron = lambda x: reduce(np.kron, x)
envinit=1.0/6.0*(np.outer(np.conj(mkron((yp,yp,ym,ym)).T), mkron((yp,yp,ym,ym)))+ \
                 np.outer(np.conj(mkron((yp,ym,ym,yp)).T), mkron((yp,ym,ym,yp)))+ \
                 np.outer(np.conj(mkron((ym,ym,yp,yp)).T), mkron((ym,ym,yp,yp)))+ \
                 np.outer(np.conj(mkron((yp,ym,yp,ym)).T), mkron((yp,ym,yp,ym)))+ \
                 np.outer(np.conj(mkron((ym,yp,ym,yp)).T), mkron((ym,yp,ym,yp)))+ \
                 np.outer(np.conj(mkron((ym,yp,yp,ym)).T), mkron((ym,yp,yp,ym))))


t=clock()
dmat=aux.dmat(ph,Omega,wlist,Alist,bmat,envinit)
t1=clock()-t; t=clock()
dmat2=aux_old.dmat(ph,Omega,wlist,Alist,bmat,envinit)
t2=clock()-t

# Check that we are consistent with Thomas' method
if np.allclose(dmat, dmat2):
    print "Success. New/Old = %.4f" % (t1/t2)
    print "Sparsity of dmat: %.3f" % (100.*np.sum(dmat==0)/np.prod(dmat.shape))
else:
    print "FAIL, new/old = %.4f" % (t1/t2)
