import numpy as np
import pauli
import auxiliary as aux
from sarmaH import *
import itertools

def main():
    """ setting parameters """
    ph=4 #number of photons
    Alist=np.array([0,0,0,0]) #hyperfine coefficients
    wlist=np.array([0,0,0,0]) #nuclear Zeeman
    bmat=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]) #dipolar coupling
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


    dmat=aux.dmat(ph,Omega,wlist,Alist,bmat,envinit)

    # tracing over the environment to obtain the reduced density matrix dmatred"""
    """
    dmatred=np.array([[0.+0.j]*2**(ph+1)]*2**(ph+1))
    envdim=2**len(Alist)
    for m in range(2**(ph+1)):
        for n in range(2**(ph+1)):
            dmatred[m,n]=np.trace(aux.dmat(ph,Omega,wlist,Alist,bmat,envinit)[m*envdim:(m+1)*envdim,n*envdim:(n+1)*envdim])"""

    #print(dmat)
    print "finished"

if __name__ == '__main__':
    from time import clock
    import anydbm
    from datetime import datetime
    from matplotlib import pyplot as plt
    starttime=clock()
    main()
    elapsedtime=clock()-starttime

    # Make a database entry and plot stuff
    now=datetime.now().ctime()
    db = anydbm.open("progress.db", "c")
    db[str(now)]=str(elapsedtime)
    everything=map(lambda x: [datetime.strptime(x[0], '%a %b %d %H:%M:%S %Y'), x[1]], db.items())
    everything=sorted(everything, key=lambda x: x[0])
    times, counts=zip(*everything)
    plt.plot_date(times, counts, '.-', lw=.5, color='black')
    plt.xlabel('Date'); plt.ylabel('Time elapsed'); plt.ylim(0,3)
    plt.savefig("progress.pdf")
