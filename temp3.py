import numpy as np 
import matplotlib.pyplot as plt 
import tikzplotlib as tpl 

from ConfSeqs import get_Gaussian_CS
from utils import Univariate_Gaussian_Source

np.random.seed(2023)


forward_CS = False 
backward_CS = True

N = 2500
nc = 500
XX = Univariate_Gaussian_Source(N=N, mean0=0.0, std0=1.0, mean1=2.0, std1=1.0,
                                nc=None, mean_vec=None, std_vec=None)

# mean_func = (np.arange(1, N+1)/N)**2
mean_func = np.zeros((N,))
# mean_func[nc:] = 1
XX += mean_func
running_mean = np.cumsum(mean_func)/np.arange(1, N+1)


if forward_CS:
    L, U = get_Gaussian_CS(XX, alpha=0.05, intersect=False)

    N_ = 700 
    compare = False 
    NN = np.arange(1, N_+1)
    ###
    legend_ = r'$\frac{1}{t}\sum_{i=1}^t \theta_t$'
    plt.plot(NN, L[:N_], color='k')
    plt.plot(NN, U[:N_], color='k', label=legend_)
    plt.plot(NN, running_mean[:N_], '--')
    if compare:
        plt.plot(NN, L[N_-1]*np.ones((N_,)), 'r-', alpha=0.2)
        plt.plot(NN, U[N_-1]*np.ones((N_,)), 'r-', alpha=0.2)
        plt.fill_between(NN, L[N_-1]*np.ones((N_,)), U[N_-1]*np.ones((N_,)), 
                        color='r', alpha=0.1)
    plt.fill_between(NN, L[:N_], U[:N_], color='gray', alpha=0.2)
    plt.ylim([-1, 1])
    plt.legend(fontsize=13)
    plt.xlabel('Number of observations (t)', fontsize=13)
    plt.ylabel('Confidence Sequence (CS)', fontsize=13)
    plt.title('Gaussian CS', fontsize=15)
    figname = 'fcs-detector-example2'
    if compare:
        figname += '_compare'
    tpl.save(figname + ".tex", axis_width=r'\figwidth', axis_height=r'\figheight')

if backward_CS:
    n1, n2, n3 = 500, 1000, 1500
    X1, X2, X3 = XX[:n1], XX[:n2], XX[:n3] 

    L1, U1 = np.flip(get_Gaussian_CS(np.flip(X1)))
    L2, U2 = np.flip(get_Gaussian_CS(np.flip(X2)))
    L3, U3 = np.flip(get_Gaussian_CS(np.flip(X3)))

    nn1 = np.arange(1, n1+1)
    nn2 = np.arange(1, n2+1)
    nn3 = np.arange(1, n3+1)
    plt.plot(nn1, L1, 'k')
    plt.plot(nn1, U1, 'k')
    plt.fill_between(nn1, L1, U1, color='gray', alpha=0.2)
    plt.plot(nn2, L2, 'k')
    plt.plot(nn2, U2, 'k')
    plt.fill_between(nn2, L2, U2, color='red', alpha=0.2)
    plt.plot(nn3, L3, 'k')
    plt.plot(nn3, U3, 'k')
    plt.fill_between(nn3, L3, U3, color='blue', alpha=0.2)
    plt.plot(nn3, running_mean[:n3], '--')
    plt.ylim([-1,1])
    plt.tick_params(left = False) #, bottom = False)
    plt.tick_params(labelleft = False) #, labelbottom = False)
    plt.title(r"Backward CSs at $n = 500, 1000, 1500$")
    figname = 'BackwardCS-example3' 
    tpl.save(figname + ".tex", axis_width=r'\figwidth', axis_height=r'\figheight')