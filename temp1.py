from utils import BoundedSource, clip_CSs 
from ConfSeqs import get_Bounded_EB_CS, get_backward_CS
import tikzplotlib as tpl 

from time import time 
import numpy as np 
import matplotlib.pyplot as plt 

# seed = int(time()%10000) 
seed = 6813
np.random.seed(seed)
print(f'The seed used is {seed}')
N = 250
nc = 100

Source = BoundedSource
Source_kwargs = {'nc':nc, 'p0':0.4, 'p1':0.7}

fwdCS = get_Bounded_EB_CS
fwdCS_kwargs = {}

X = Source(N=N, **Source_kwargs)

n1, n2, n3 = int(0.9*nc), int(1.5*nc), int(0.95*N)

def func(X, n, CSfunc, CSkwargs, alpha=0.01, nc=400, 
            save_fig=False, figname=None): 
    X_ = X[:n]
    CS = CSfunc(X_, **CSkwargs)
    backCS = get_backward_CS(X_, CSfunc, CSkwargs, alpha=alpha,
                                    two_sample_data=False)
    L, U = CS
    Lb, Ub = backCS

    L, U, Lb, Ub = clip_CSs(L, U, Lb, Ub, amin=0, amax=1)
    nn = np.arange(1, n+1)
    plt.figure() 
    if n<=nc:
        plt.plot(nn, X_, 'ko', alpha=0.3) 
    else: 
        prechangeX, postchangeX = X_[:nc], X_[nc:]
        nn1, nn2 = np.arange(1, nc+1), np.arange(nc+1, n+1) 
        plt.plot(nn1, prechangeX, 'ko', alpha=0.3)
        plt.plot(nn2, postchangeX, 'rd', alpha=0.3)
    # plt.plot(nn, L, 'k')
    # plt.plot(nn, U, 'k')
    # plt.plot(nn, Lb, 'r')
    # plt.plot(nn, Ub, 'r')
    plt.fill_between(nn, L, U, alpha=0.2, color='gray')
    plt.fill_between(nn, Lb, Ub, alpha=0.2, color='red')
    plt.tick_params(left = False) #, bottom = False)
    plt.tick_params(labelleft = False) #, labelbottom = False)
    if save_fig: 
        figname = 'temp' if figname is None else figname 
        plt.savefig(figname+'.png', dpi=450, bbox_inches='tight')
        tpl.save(figname + ".tex", axis_width=r'\figwidth', axis_height=r'\figheight')
    plt.show()


save_fig = True
figname1 = './data/illustration1'
figname2 = './data/illustration2'
figname3 = './data/illustration3'
func(X, n1, fwdCS, fwdCS_kwargs, nc=nc, save_fig=save_fig, figname=figname1)
func(X, n2, fwdCS, fwdCS_kwargs, nc=nc, save_fig=save_fig, figname=figname2)
func(X, n3, fwdCS, fwdCS_kwargs, nc=nc, save_fig=save_fig, figname=figname3)
