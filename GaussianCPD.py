import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt

from changepoint import CPD_strategy1, CPD_strategy2 
from utils import get_Gaussian_CS, Univariate_Gaussian_Source, get_backward_Gaussian_CS
from utils import check_disjoint_univariate as check_disjoint1
from utils import check_disjoint_univariate_two_CS as check_disjoint2


N = 2000 
nc = 250 
alpha = 0.004

fwdCS_func = get_Gaussian_CS 
def backCS_func(XX, alpha, t, intersect=False):
    XX_ = XX[:t] 
    return get_backward_Gaussian_CS(XX=XX_, alpha=alpha, intersect=intersect)

XX = Univariate_Gaussian_Source(mean0=0, mean1=0.0, N=N, nc=nc)

strategy1=True
strategy2=True

if strategy1:

    CS, stopped, stopping_time  = CPD_strategy1(
        XX=XX, tmax=len(XX), CS_func=fwdCS_func, CS_kwargs={},
        check_disjoint=check_disjoint1, alpha=alpha, 
        plot_fig=False
    )

    print(stopping_time, stopped)

    L, U = CS 
    NN = np.arange(1, len(U)+1)
    plt.plot(NN, L)
    plt.plot(NN, U)
    plt.title('Strategy 1')
    plt.show()

if strategy2: 
    fwdCS, backCS, stopped, stopping_time = CPD_strategy2(
        XX=XX, tmax=len(XX), fwdCS_func=fwdCS_func, fwdCS_kwargs={}, 
        backCS_func=backCS_func, backCS_kwargs={}, check_disjoint=check_disjoint2, 
        alpha=alpha
    )

    L, U = fwdCS 
    L_, U_ = backCS 
    NN = np.arange(1, len(U)+1)
    NN_ = np.arange(1, len(U_)+1)
    plt.plot(NN, L, color='r')
    plt.plot(NN, U, color='r')
    plt.plot(NN_, L_, color='b')
    plt.plot(NN_, U_, color='b')
    plt.title('Strategy 2')
    plt.show()

