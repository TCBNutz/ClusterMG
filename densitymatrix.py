import numpy as np
import pauli
import auxiliary as aux
import auxiliary_old as aux_old
from sarmaH import *
import multiprocessing

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

def getdm_new(x): return aux.dmat(ph,Omega,wlist,Alist,bmat,envinit)

def getdm_old(x): return aux_old.dmat(ph,Omega,wlist,Alist,bmat,envinit)

if __name__ == '__main__':
    iterations=8

    # Compare some speeds
    print "Old method, %d iterations, using 1 CPU..." % (iterations)
    traces=map(getdm_old, range(iterations))
    print "Done.\n"

    print "New method, %d iterations, using 1 CPU..." % (iterations)
    dms=map(getdm_new, range(iterations))
    print "Done.\n"

    cpus=multiprocessing.cpu_count()
    p=multiprocessing.Pool(cpus)
    print "New method, %d interations, using %d CPUs..." % (iterations, cpus)
    dms=p.map(getdm_new, range(iterations))
    print "Done.\n"

    # Check that we are consistent with Thomas' method
    print "Checking consistency..."
    dmat=getdm_new(0); dmat2=getdm_new(0)
    if np.allclose(dmat, dmat2):
        print "SUCCESS"
    else:
        print "FAIL" 
